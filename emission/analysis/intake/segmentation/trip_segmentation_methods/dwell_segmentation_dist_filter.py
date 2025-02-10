from __future__ import division, unicode_literals, print_function, absolute_import
# Standard imports
from future import standard_library
standard_library.install_aliases()
from builtins import str, range
from past.utils import old_div
import logging
import attrdict as ad
import numpy as np
import pandas as pd
import time

# Our imports
import emission.analysis.point_features as pf
import emission.analysis.intake.segmentation.trip_segmentation as eaist
import emission.core.wrapper.location as ecwl
import emission.analysis.intake.segmentation.restart_checking as eaisr
import emission.analysis.intake.segmentation.trip_segmentation_methods.trip_end_detection_corner_cases as eaistc
import emission.storage.decorations.stats_queries as esds
import emission.core.timer as ect
import emission.core.wrapper.pipelinestate as ecwp

# A helper “haversine” function (vectorized) – same as used in the time‐filter.
def haversine(lon1, lat1, lon2, lat2):
    earth_radius = 6371000  # meters
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    lon1, lon2 = np.radians(lon1), np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * earth_radius * np.arcsin(np.sqrt(a))

TWELVE_HOURS = 12 * 60 * 60

class DwellSegmentationDistFilter(eaist.TripSegmentationMethod):
    def __init__(self, time_threshold, point_threshold, distance_threshold):
        """
        Determines segmentation points for points that were generated using a
        distance filter (i.e. report points every n meters). This will *not* work for
        points generated using a time filter, because it expects to have a
        time gap between subsequent points to detect the trip end.
        
        On iOS, we sometimes get points even when the phone is not in motion,
        triggered by zigzagging between low quality points.
        """
        self.time_threshold = time_threshold
        self.point_threshold = point_threshold
        self.distance_threshold = distance_threshold

    def segment_into_trips(self, timeseries, time_query, filtered_points_df):
        """
        Returns the segmentation points (as start and end point pairs) for a range of data.
        The logic is the same as before, but much of the per‐row math is done
        in vectorized numpy calls.
        """
        with ect.Timer() as t_get:
            # Work on a copy with a reset index.
            df = filtered_points_df.copy().reset_index(drop=True)
            user_id = df["user_id"].iloc[0]
        esds.store_pipeline_time(
            user_id,
            ecwp.PipelineStages.TRIP_SEGMENTATION.name + "/segment_into_trips_dist/get_filtered_points_df",
            time.time(),
            t_get.elapsed
        )

        # Mark every point as valid initially.
        df["valid"] = True

        # Retrieve auxiliary data.
        transition_df = timeseries.get_data_df("statemachine/transition", time_query)
        # For motion activities we try to get a dataframe (as in the time filter)
        motion_df = timeseries.get_data_df("background/motion_activity", time_query)

        # Precompute arrays for latitude, longitude, timestamp and metadata_write_ts.
        lat = df["latitude"].to_numpy()
        lon = df["longitude"].to_numpy()
        ts = df["ts"].to_numpy()
        meta_ts = df["metadata_write_ts"].to_numpy()
        n = len(df)
        if n == 0:
            return []

        # Compute the time gap between consecutive points.
        delta_time = np.zeros(n)
        delta_time[1:] = ts[1:] - ts[:-1]

        # Compute the distance between consecutive points using our vectorized haversine.
        delta_distance = np.zeros(n)
        delta_distance[1:] = haversine(lon[:-1], lat[:-1], lon[1:], lat[1:])

        # Compute speed (distance per time). Avoid division by zero.
        speed = np.zeros(n)
        with np.errstate(divide='ignore', invalid='ignore'):
            speed[1:] = np.where(delta_time[1:] > 0, delta_distance[1:] / delta_time[1:], 0)

        # For “glomming” points that occur right after a detected trip end,
        # we use the condition that the gap from the previous point is small.
        continue_mask = np.zeros(n, dtype=bool)
        continue_mask[1:] = delta_distance[1:] < self.distance_threshold

        segmentation_indices = []
        trip_start_idx = 0
        just_ended = True  # We begin in the “just ended” state so that we can decide on a start.
        self.last_ts_processed = None

        with ect.Timer() as t_loop:
            i = 0
            while i < n:
                if just_ended:
                    # When we just ended a trip, some extra points might still be
                    # part of the previous trip if they are within the distance threshold.
                    if i > 0 and continue_mask[i]:
                        self.last_ts_processed = meta_ts[i]
                        i += 1
                        continue
                    else:
                        # Otherwise, start a new trip here.
                        trip_start_idx = i
                        just_ended = False
                        i += 1
                        continue
                else:
                    # In an ongoing trip we compare the current point to the last valid point.
                    last_valid_idx = i - 1  # normally the previous point is valid.
                    dt = ts[i] - ts[last_valid_idx]
                    dd = haversine(lon[last_valid_idx], lat[last_valid_idx], lon[i], lat[i])
                    sp = dd / dt if dt > 0 else np.nan
                    # Compute a speed threshold from the distance and time thresholds.
                    speed_threshold = (4 * self.distance_threshold) / self.time_threshold

                    if dt > self.time_threshold:
                        # (1) If tracking was restarted in this time window, end the trip.
                        if eaisr.is_tracking_restarted_in_range(ts[last_valid_idx], ts[i],
                                                                 timeseries, transition_df):
                            segmentation_indices.append((trip_start_idx, last_valid_idx))
                            self.last_ts_processed = meta_ts[i]
                            just_ended = True
                            i += 1
                            continue

                        # (2) If there was no ongoing motion during the gap, end the trip.
                        ongoing_motion = (len(eaisr.get_ongoing_motion_in_range(ts[last_valid_idx], ts[i],
                                                                                timeseries, motion_df)) > 0)
                        if dt > self.time_threshold and (not ongoing_motion):
                            segmentation_indices.append((trip_start_idx, last_valid_idx))
                            self.last_ts_processed = meta_ts[i]
                            just_ended = True
                            i += 1
                            continue

                        # (3) If the gap is huge (e.g. >12 hours), end the trip.
                        if dt > TWELVE_HOURS:
                            segmentation_indices.append((trip_start_idx, last_valid_idx))
                            self.last_ts_processed = meta_ts[i]
                            just_ended = True
                            i += 1
                            continue

                        # (4) If we’ve been here a while but haven’t moved much, end the trip.
                        if dt > self.time_threshold and sp < speed_threshold:
                            # Before ending the trip, check for a huge (and invalid) timestamp offset.
                            lastPoint = ad.AttrDict(df.loc[last_valid_idx])
                            currPoint = ad.AttrDict(df.loc[i])
                            ongoing_motion_range = eaisr.get_ongoing_motion_in_range(ts[last_valid_idx], ts[i],
                                                                                      timeseries, motion_df)
                            if eaistc.is_huge_invalid_ts_offset(self, lastPoint, currPoint,
                                                                 timeseries, ongoing_motion_range):
                                # Mark this spurious point as invalid.
                                df.at[i, "valid"] = False
                                timeseries.invalidate_raw_entry(currPoint["_id"])
                                i += 1
                                continue
                            else:
                                segmentation_indices.append((trip_start_idx, last_valid_idx))
                                self.last_ts_processed = meta_ts[i]
                                just_ended = True
                                i += 1
                                continue
                    # If none of the above conditions hold, keep going.
                    i += 1

            # Finally, if we ended the loop in an “ongoing trip” state,
            # we may want to mark the last trip as ended if there is evidence from transitions.
            if (not just_ended) and (n > 0):
                last_point_ts = ts[-1]
                stopped_moving_after_last = transition_df[
                    (transition_df.ts > last_point_ts) & (transition_df.transition == 2)
                ]
                if len(stopped_moving_after_last) > 0:
                    segmentation_indices.append((trip_start_idx, n - 1))
                    self.last_ts_processed = meta_ts[-1]
        esds.store_pipeline_time(
            user_id,
            ecwp.PipelineStages.TRIP_SEGMENTATION.name + "/segment_into_trips_dist/loop",
            time.time(),
            t_loop.elapsed
        )

        # Convert segmentation indices back into (start, end) AttrDict pairs.
        segmentation_points = []
        for start_idx, end_idx in segmentation_indices:
            start_point = ad.AttrDict(df.loc[start_idx])
            end_point = ad.AttrDict(df.loc[end_idx])
            segmentation_points.append((start_point, end_point))
            logging.info(f"Segmented trip: {start_point.fmt_time} -> {end_point.fmt_time}")

        return segmentation_points
