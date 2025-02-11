from __future__ import division, unicode_literals, print_function, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import str
from past.utils import old_div
import logging
import attrdict as ad
import numpy as np
import pandas as pd
import datetime as pydt
import time

# Our imports
import emission.analysis.point_features as pf
import emission.analysis.intake.segmentation.trip_segmentation as eaist
import emission.core.wrapper.location as ecwl
import emission.analysis.intake.segmentation.restart_checking as eaisr
import emission.storage.decorations.stats_queries as esds
import emission.core.timer as ect
import emission.core.wrapper.pipelinestate as ecwp

TWELVE_HOURS = 12 * 60 * 60

def haversine(lon1, lat1, lon2, lat2):
    earth_radius = 6371000  # meters
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    lon1, lon2 = np.radians(lon1), np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    return 2 * earth_radius * np.arcsin(np.sqrt(a))

class DwellSegmentationDistFilter(eaist.TripSegmentationMethod):
    def __init__(self, time_threshold, point_threshold, distance_threshold):
        """
        Determines segmentation points for points that were generated using a
        distance filter. This version uses an iterative sliding-window approach
        (see compute_recent_point_diffs_optimized) to reduce overhead compared to
        DataFrame.apply-based methods.
        """
        self.time_threshold = time_threshold
        self.point_threshold = point_threshold
        self.distance_threshold = distance_threshold

    def segment_into_trips(self, timeseries, time_query, filtered_points_df):
        user_id = filtered_points_df["user_id"].iloc[0]
        # Reset the index to allow integer-based indexing
        filtered_points_df.reset_index(inplace=True)

        self.transition_df = timeseries.get_data_df("statemachine/transition", time_query)
        self.motion_df = timeseries.get_data_df("background/motion_activity", time_query)
        self.last_ts_processed = None
        logging.info("Last ts processed = %s", self.last_ts_processed)

        # Use the optimized recent-points differences calculation.
        filtered_points_df['recent_points_diffs'] = self.compute_recent_point_diffs_optimized(filtered_points_df)

        # Compute last diff values without using .apply lambdas.
        dist_diff = []
        ts_diff = []
        for diffs in filtered_points_df['recent_points_diffs']:
            if diffs.shape[1] > 0:
                dist_diff.append(diffs[0, -1])
                ts_diff.append(diffs[1, -1])
            else:
                dist_diff.append(np.nan)
                ts_diff.append(np.nan)
        filtered_points_df['dist_diff'] = dist_diff
        filtered_points_df['ts_diff'] = ts_diff
        filtered_points_df['speed_diff'] = filtered_points_df['dist_diff'] / filtered_points_df['ts_diff']

        # These columns are used to decide if a new trip has started.
        filtered_points_df['ongoing_motion'] = eaisr.ongoing_motion_in_loc_df(filtered_points_df, self.motion_df)
        filtered_points_df['tracking_restarted'] = eaisr.tracking_restarted_in_loc_df(filtered_points_df, self.transition_df)

        segmentation_idx_pairs = []
        trip_start_idx = 0

        with ect.Timer() as t_loop:
            while trip_start_idx < len(filtered_points_df):
                logging.info("trip_start_idx = %d", trip_start_idx)
                # For the distance filter, a jump beyond the threshold suggests a trip end.
                potential_trip_end_idxs = np.where(
                    (filtered_points_df.index > trip_start_idx) &
                    (filtered_points_df['dist_diff'] > self.distance_threshold)
                )[0]

                logging.info("potential_trip_end_idxs = %s", potential_trip_end_idxs)
                if len(potential_trip_end_idxs) == 0:
                    logging.info("No more segments found starting from index %d", trip_start_idx)
                    trip_start_idx = len(filtered_points_df)
                    break

                trip_end_detected_idx = potential_trip_end_idxs[0]
                logging.info("Trip end detected at index %d", trip_end_detected_idx)

                ended_before_this, trip_end_idx = self.get_last_trip_end_point_idx(
                    trip_end_detected_idx,
                    filtered_points_df.iloc[trip_end_detected_idx]['recent_points_diffs']
                )
                segmentation_idx_pairs.append((trip_start_idx, trip_end_idx))

                if ended_before_this:
                    trip_start_idx = trip_end_detected_idx
                    self.last_ts_processed = float(filtered_points_df.iloc[trip_start_idx]['metadata_write_ts'])
                    logging.info("Setting new trip start index to %d", trip_start_idx)
                else:
                    next_start_idxs = filtered_points_df[
                        (filtered_points_df.index > trip_end_detected_idx) &
                        ((filtered_points_df['ts_diff'] > 60) |
                         (filtered_points_df['dist_diff'] >= self.distance_threshold))
                    ].index
                    if len(next_start_idxs) > 0:
                        trip_start_idx = next_start_idxs[0]
                        self.last_ts_processed = float(filtered_points_df.iloc[trip_start_idx - 1]['metadata_write_ts'])
                        logging.info("Setting new trip start index to %d", trip_start_idx)
                    elif trip_end_detected_idx + 1 < len(filtered_points_df):
                        trip_start_idx = trip_end_detected_idx + 1
                        self.last_ts_processed = float(filtered_points_df.iloc[trip_start_idx]['metadata_write_ts'])
                        logging.info("Setting new trip start index to %d", trip_start_idx)
                    else:
                        trip_start_idx = len(filtered_points_df)
                        self.last_ts_processed = float(filtered_points_df.iloc[-1]['metadata_write_ts'])
                        logging.info("Setting new trip start index to end of dataframe")

        esds.store_pipeline_time(
            user_id,
            ecwp.PipelineStages.TRIP_SEGMENTATION.name + "/segment_into_trips_dist/loop",
            time.time(),
            t_loop.elapsed
        )

        segmentation_points = [
            (ad.AttrDict(filtered_points_df.iloc[start_idx]),
             ad.AttrDict(filtered_points_df.iloc[end_idx]))
            for (start_idx, end_idx) in segmentation_idx_pairs
        ]
        logging.info("self.last_ts_processed = %s", self.last_ts_processed)
        for (p1, p2) in segmentation_points:
            logging.info("%s, %s -> %s, %s", p1.get("index"), p1.get("ts"), p2.get("index"), p2.get("ts"))
        return segmentation_points

    def compute_recent_point_diffs_optimized(self, df):
        """
        Optimized computation of recent differences using an iterative loop.
        """
        timestamps = df["ts"].to_numpy()
        lat = df["latitude"].to_numpy()
        lon = df["longitude"].to_numpy()
        N = len(df)
        diffs = [None] * N
        for i in range(N):
            start_index = max(0, i - self.point_threshold)
            while start_index < i and timestamps[start_index] < timestamps[i] - self.time_threshold:
                start_index += 1
            window_length = i - start_index
            if window_length > 0:
                dists = haversine(lon[start_index:i], lat[start_index:i], lon[i], lat[i])
                dt = timestamps[i] - timestamps[start_index:i]
                diffs[i] = np.vstack((dists, dt))
            else:
                diffs[i] = np.empty((2, 0))
        return pd.Series(diffs, index=df.index)

    def get_last_trip_end_point_idx(self, curr_idx, recent_diffs: np.ndarray):
        """
        Determines the best candidate for the trip end point from the recent diffs.
        (Logic retained from your time filter segmentation code.)
        """
        if recent_diffs.shape[1] == 0:
            return (False, curr_idx)
        recent_diffs_non_na = recent_diffs[:, ~np.isnan(recent_diffs[0, :])]
        num_recent_diffs_in_point_threshold = min(len(recent_diffs_non_na[0, :]), self.point_threshold)
        num_recent_diffs_in_time_threshold = np.sum(recent_diffs_non_na[1, :] < self.time_threshold)
        ended_before_this = (num_recent_diffs_in_time_threshold == 0)
        last_n_median_idx = np.median(np.arange(curr_idx - num_recent_diffs_in_point_threshold, curr_idx + 1))
        if ended_before_this:
            last_trip_end_index = int(last_n_median_idx)
        else:
            last_time_median_idx = np.median(np.arange(curr_idx - num_recent_diffs_in_time_threshold, curr_idx))
            last_trip_end_index = int(min(last_time_median_idx, last_n_median_idx))
        logging.debug("curr_idx = %d, num_recent_diffs_in_point_threshold = %d, num_recent_diffs_in_time_threshold = %d, ended_before_this = %s, last_trip_end_index = %d",
                      curr_idx, num_recent_diffs_in_point_threshold, num_recent_diffs_in_time_threshold, ended_before_this, last_trip_end_index)
        return (ended_before_this, last_trip_end_index)
