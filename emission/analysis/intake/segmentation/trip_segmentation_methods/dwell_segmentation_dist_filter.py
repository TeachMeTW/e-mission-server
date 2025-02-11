from __future__ import division, unicode_literals, print_function, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import str
from past.utils import old_div
import logging
import attrdict as ad
import numpy as np
import datetime as pydt
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

class DwellSegmentationDistFilter(eaist.TripSegmentationMethod):
    def __init__(self, time_threshold, point_threshold, distance_threshold):
        """
        Determines segmentation points for points that were generated using a
        distance filter (i.e. report points every n meters). At least on iOS,
        we sometimes get points even when the phone is not in motion. This seems
        to be triggered by zigzagging between low quality points.
        """
        self.time_threshold = time_threshold
        self.point_threshold = point_threshold
        self.distance_threshold = distance_threshold
        # Pre-compute approximate speed threshold:
        # speedThreshold = (distance_threshold * 2) / (time_threshold/2) = 4 * distance_threshold / time_threshold
        self.speedThreshold = 4.0 * self.distance_threshold / self.time_threshold

    def segment_into_trips(self, timeseries, time_query, filtered_points_df):
        """
        Process the filtered points (assumed to be generated using a distance filter)
        and return a list of (trip_start, trip_end) segmentation tuples.
        """
        with ect.Timer() as t_get_filtered_points:
            self.filtered_points_df = filtered_points_df
            user_id = self.filtered_points_df["user_id"].iloc[0]
        esds.store_pipeline_time(
            user_id,
            ecwp.PipelineStages.TRIP_SEGMENTATION.name + "/segment_into_trips_dist/get_filtered_points_df",
            time.time(),
            t_get_filtered_points.elapsed
        )

        # Mark all points as valid initially.
        self.filtered_points_df.loc[:, "valid"] = True

        # Retrieve state machine transitions and motion activities.
        self.transition_df = timeseries.get_data_df("statemachine/transition", time_query)
        self.motion_list = list(timeseries.find_entries(["background/motion_activity"], time_query))
        if len(self.transition_df) > 0:
            logging.debug("self.transition_df = %s", self.transition_df[["fmt_time", "transition"]])
        else:
            logging.debug("no transitions found. This can happen for continuous sensing")

        segmentation_points = []
        self.last_ts_processed = None
        logging.info("Last ts processed = %s", self.last_ts_processed)

        df = self.filtered_points_df  # local alias for speed
        n_points = len(df)
        i = 0
        just_ended = True
        curr_trip_start_point = None

        with ect.Timer() as t_loop:
            while i < n_points:
                currPoint = ad.AttrDict(df.iloc[i])
                currPoint.idx = i
                logging.debug("Processing point idx=%s, time=%s", i, currPoint.fmt_time)

                if just_ended:
                    # Check whether the current point should merge with the previous trip.
                    if self.continue_just_ended(i, currPoint, df):
                        self.last_ts_processed = currPoint.metadata_write_ts
                        i += 1
                        continue
                    else:
                        # Start a new trip segment.
                        curr_trip_start_point = currPoint
                        just_ended = False
                        i += 1
                        continue

                # We are inside an ongoing trip; scan ahead until a trip end is detected.
                trip_ended = False
                j = i
                while j < n_points and not trip_ended:
                    currPoint = ad.AttrDict(df.iloc[j])
                    currPoint.idx = j
                    last_valid_point = self.find_last_valid_point(j)
                    if self.has_trip_ended(last_valid_point, currPoint, timeseries):
                        trip_ended = True
                        break
                    j += 1

                if trip_ended:
                    # End the current trip segment at the last valid point.
                    final_valid = self.find_last_valid_point(j)
                    segmentation_points.append((curr_trip_start_point, final_valid))
                    logging.info("Found trip end at %s", final_valid.fmt_time)
                    self.last_ts_processed = currPoint.metadata_write_ts
                    just_ended = True
                    # Possibly start a new trip from the current point.
                    if not self.continue_just_ended(j, currPoint, df):
                        curr_trip_start_point = currPoint
                        just_ended = False
                    i = j + 1
                else:
                    # No trip end was detected; we have reached the end.
                    i = j

        esds.store_pipeline_time(
            user_id,
            ecwp.PipelineStages.TRIP_SEGMENTATION.name + "/segment_into_trips_dist/loop",
            time.time(),
            t_loop.elapsed
        )

        # Check for a possible incomplete final trip.
        if not just_ended and n_points > 0:
            currPoint = ad.AttrDict(df.iloc[-1])
            if len(self.transition_df) > 0:
                stopped_moving_after_last = self.transition_df[
                    (self.transition_df.ts > currPoint.ts) &
                    (self.transition_df.transition == 2)
                ]
                logging.debug("stopped_moving_after_last = %s", stopped_moving_after_last[["fmt_time", "transition"]])
                if len(stopped_moving_after_last) > 0:
                    logging.debug("Found %d transitions after last point, ending trip...", len(stopped_moving_after_last))
                    segmentation_points.append((curr_trip_start_point, currPoint))
                    self.last_ts_processed = currPoint.metadata_write_ts
                else:
                    logging.debug("Found %d transitions after last point, not ending trip...", len(stopped_moving_after_last))
        return segmentation_points

    def has_trip_ended(self, lastPoint, currPoint, timeseries):
        timeDelta = currPoint.ts - lastPoint.ts
        distDelta = pf.calDistance(lastPoint, currPoint)
        logging.debug("lastPoint = %s, time difference = %s, dist difference = %s",
                      lastPoint, timeDelta, distDelta)
        if timeDelta > self.time_threshold:
            speedDelta = old_div(distDelta, timeDelta) if timeDelta > 0 else np.nan
            if eaisr.is_tracking_restarted_in_range(lastPoint.ts, currPoint.ts, timeseries, self.transition_df):
                logging.debug("Tracking was restarted, ending trip")
                return True
            ongoing_motion_in_range = eaisr.get_ongoing_motion_in_range(lastPoint.ts, currPoint.ts, timeseries, self.motion_list)
            if timeDelta > self.time_threshold and not ongoing_motion_in_range:
                logging.debug("Large gap (%s > %s) with no ongoing motion; ending trip",
                              timeDelta, self.time_threshold)
                return True
            TWELVE_HOURS = 12 * 60 * 60
            if timeDelta > TWELVE_HOURS:
                logging.debug("Time gap > 12 hours; ending trip")
                return True
            if timeDelta > self.time_threshold and speedDelta < self.speedThreshold:
                if eaistc.is_huge_invalid_ts_offset(self, lastPoint, currPoint, timeseries, ongoing_motion_in_range):
                    logging.debug("Invalid timestamp offset detected for point idx %s", currPoint.idx)
                    self.filtered_points_df.valid.iloc[currPoint.idx] = False
                    timeseries.invalidate_raw_entry(currPoint["_id"])
                    return False
                else:
                    logging.debug("Ending trip due to large time gap and low speed")
                    return True
            else:
                logging.debug("Continuing trip: time gap %s vs %s, dist gap %s vs %s, speed gap %s vs %s",
                              timeDelta, self.time_threshold, distDelta, self.distance_threshold,
                              speedDelta, self.speedThreshold)
                return False
        else:
            return False

    def find_last_valid_point(self, idx):
        lastPoint = ad.AttrDict(self.filtered_points_df.iloc[idx - 1])
        if lastPoint.valid:
            return lastPoint
        i = 2
        while not lastPoint.valid and (idx - i) >= 0:
            lastPoint = ad.AttrDict(self.filtered_points_df.iloc[idx - i])
            i += 1
        return lastPoint

    def continue_just_ended(self, idx, currPoint, filtered_points_df):
        """
        Sometimes a point that occurs just after a trip end (e.g. within a minute
        and within the distance threshold) should be considered as belonging to
        the previous trip.
        """
        if idx == 0:
            return False
        else:
            lastPoint = ad.AttrDict(filtered_points_df.iloc[idx - 1])
            deltaDist = pf.calDistance(lastPoint, currPoint)
            deltaTime = currPoint.ts - lastPoint.ts
            logging.debug("Comparing with lastPoint = %s, distance = %s (< %s), time = %s (< %s)",
                          lastPoint, deltaDist, self.distance_threshold, deltaTime, self.time_threshold)
            if deltaDist < self.distance_threshold:
                logging.info("Points %s and %s are %d apart (within distance threshold); merging into same trip",
                             lastPoint["_id"], currPoint["_id"], deltaDist)
                return True
            else:
                return False
