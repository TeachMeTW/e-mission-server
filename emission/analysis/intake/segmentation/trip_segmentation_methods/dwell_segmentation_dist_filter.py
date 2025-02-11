from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
# Standard imports
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

# A useful constant
TWELVE_HOURS = 12 * 60 * 60

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance (in meters) between two points
    on the earth (specified in decimal degrees).
    """
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
        distance filter (i.e. report points every n meters). Originally, the
        algorithm iterated over points one by one. This version has been modified
        to use a vectorized, window-based approach similar to that used in the
        time filter segmentation method.

        At least on iOS, we sometimes get points even when the phone is not in
        motion. This seems to be triggered by zigzagging between low quality points.
        """
        self.time_threshold = time_threshold
        self.point_threshold = point_threshold
        self.distance_threshold = distance_threshold

    def segment_into_trips(self, timeseries, time_query, filtered_points_df):
        """
        Examines the timeseries database for a specific range and returns the
        segmentation points. This vectorized implementation is modeled on the
        time filter segmentation method, but uses the parameters (including a
        distance threshold) appropriate for points generated via a distance filter.
        """
        user_id = filtered_points_df["user_id"].iloc[0]
        # Reset the index so that we can use integer indexing throughout.
        filtered_points_df.reset_index(inplace=True)

        # Retrieve transition and motion data
        self.transition_df = timeseries.get_data_df("statemachine/transition", time_query)
        self.motion_df = timeseries.get_data_df("background/motion_activity", time_query)
        if len(self.transition_df) > 0:
            logging.debug("self.transition_df = %s", self.transition_df[["fmt_time", "transition"]])
        else:
            logging.debug("no transitions found. This can happen for continuous sensing")

        self.last_ts_processed = None
        logging.info("Last ts processed = %s", self.last_ts_processed)

        # Compute a sliding window of recent distance and time differences.
        filtered_points_df['recent_points_diffs'] = self.compute_recent_point_diffs(filtered_points_df)
        # For convenience, extract the last (most recent) distance and time difference.
        filtered_points_df['dist_diff'] = filtered_points_df['recent_points_diffs'].apply(
            lambda x: x[0, -1] if x.shape[1] > 0 else np.nan)
        filtered_points_df['ts_diff'] = filtered_points_df['recent_points_diffs'].apply(
            lambda x: x[1, -1] if x.shape[1] > 0 else np.nan)
        filtered_points_df['speed_diff'] = filtered_points_df['dist_diff'] / filtered_points_df['ts_diff']

        # Compute additional columns to help decide whether a new trip has started.
        filtered_points_df['ongoing_motion'] = eaisr.ongoing_motion_in_loc_df(filtered_points_df, self.motion_df)
        filtered_points_df['tracking_restarted'] = eaisr.tracking_restarted_in_loc_df(filtered_points_df, self.transition_df)

        segmentation_idx_pairs = []
        trip_start_idx = 0

        with ect.Timer() as t_loop:
            while trip_start_idx < len(filtered_points_df):
                logging.info("trip_start_idx = %d", trip_start_idx)

                # For each point, compute the maximum distance difference within the recent window.
                recent_diffs = filtered_points_df['recent_points_diffs']
                max_recent_dist_diffs = recent_diffs.apply(
                    lambda diffs: np.nan if diffs.shape[1] < self.point_threshold - 2 
                    else diffs[0, :].max()
                )

                # Identify potential trip end indices based on several conditions:
                #   - A tracking restart was detected.
                #   - A long time gap with little motion.
                #   - A very long gap (e.g. > 12 hours).
                #   - A long gap and low speed.
                #   - A jump in distance that exceeds the configured threshold.
                potential_trip_end_idxs = np.where(
                    (filtered_points_df.index > trip_start_idx) & (
                        (filtered_points_df['tracking_restarted']) |
                        ((filtered_points_df['ts_diff'] > 2 * self.time_threshold) & (~filtered_points_df['ongoing_motion'])) |
                        (filtered_points_df['ts_diff'] > TWELVE_HOURS) |
                        ((filtered_points_df['ts_diff'] > 2 * self.time_threshold) &
                         (filtered_points_df['speed_diff'] < (self.distance_threshold / self.time_threshold))) |
                        (filtered_points_df['dist_diff'] > self.distance_threshold)
                    )
                )[0]

                logging.info("potential_trip_end_idxs = %s", potential_trip_end_idxs)

                if len(potential_trip_end_idxs) == 0:
                    logging.info("No more segments found starting from index %d", trip_start_idx)
                    # If there is a transition after the last point, mark the trip end.
                    if trip_start_idx < len(filtered_points_df) - 1 and len(self.transition_df) > 0:
                        last_point_ts = filtered_points_df.iloc[-1]['ts']
                        stopped_moving_after_last = self.transition_df[
                            (self.transition_df['ts'] > last_point_ts) &
                            (self.transition_df['transition'] == 2)
                        ]
                        logging.info("Stopped moving after last point: %s",
                                     stopped_moving_after_last[["fmt_time", "transition"]])
                        if len(stopped_moving_after_last) > 0:
                            (_, trip_end_idx) = self.get_last_trip_end_point_idx(
                                len(filtered_points_df) - 1,
                                filtered_points_df.iloc[-1]['recent_points_diffs']
                            )
                            segmentation_idx_pairs.append((trip_start_idx, trip_end_idx))
                            logging.info("Found trip end at index %d", trip_end_idx)
                            self.last_ts_processed = float(filtered_points_df.iloc[-1]['metadata_write_ts'])
                    trip_start_idx = len(filtered_points_df)
                    break

                trip_end_detected_idx = potential_trip_end_idxs[0]
                logging.info("***** TRIP END DETECTED AT index %d of %d *****",
                             trip_end_detected_idx, len(filtered_points_df) - 1)
                logging.debug("recent_points_diffs at detected index: %s",
                              filtered_points_df.iloc[trip_end_detected_idx]['recent_points_diffs'])

                # Use a helper function (adapted from the time filter) to choose the best trip end index.
                ended_before_this, trip_end_idx = self.get_last_trip_end_point_idx(
                    trip_end_detected_idx,
                    filtered_points_df.iloc[trip_end_detected_idx]['recent_points_diffs']
                )
                segmentation_idx_pairs.append((trip_start_idx, trip_end_idx))

                if ended_before_this:
                    # A significant gap before the detected index indicates that this point
                    # should be the start of the new trip.
                    trip_start_idx = trip_end_detected_idx
                    self.last_ts_processed = float(filtered_points_df.iloc[trip_start_idx]['metadata_write_ts'])
                    logging.info("Setting new trip start index to %d", trip_start_idx)
                else:
                    # Look for the next point that is sufficiently separated (in time or distance)
                    # to mark the beginning of the next trip.
                    next_start_idxs = filtered_points_df[
                        (filtered_points_df.index > trip_end_detected_idx) & (
                            (filtered_points_df['ts_diff'] > 60) |
                            (filtered_points_df['dist_diff'] >= self.distance_threshold)
                        )
                    ].index
                    if len(next_start_idxs) > 0:
                        trip_start_idx = next_start_idxs[0]
                        logging.info("Setting new trip start index to %d", trip_start_idx)
                        self.last_ts_processed = float(filtered_points_df.iloc[trip_start_idx - 1]['metadata_write_ts'])
                    elif trip_end_detected_idx + 1 < len(filtered_points_df):
                        trip_start_idx = trip_end_detected_idx + 1
                        logging.info("Setting new trip start index to %d", trip_start_idx)
                        self.last_ts_processed = float(filtered_points_df.iloc[trip_start_idx]['metadata_write_ts'])
                    else:
                        trip_start_idx = len(filtered_points_df)
                        logging.info("Setting new trip start index to end of dataframe")
                        self.last_ts_processed = float(filtered_points_df.iloc[-1]['metadata_write_ts'])

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

    def compute_recent_point_diffs(self, df):
        """
        For each point, compute the recent differences in distance and time relative to a
        starting point defined by the smaller of the point threshold and the time threshold.
        Returns a pandas Series where each element is a numpy array of shape (2, N) with:
          - row 0: the distances (in meters) computed using the haversine function,
          - row 1: the time differences (in seconds).
        """
        indices = df.index.to_numpy()
        # Determine start indices based on the point threshold and time threshold.
        last_n_start_indices = np.searchsorted(indices, indices - self.point_threshold)
        timestamps = df["ts"].to_numpy()
        last_time_start_indices = np.searchsorted(timestamps, timestamps - self.time_threshold, side='right')
        start_indices = [min(a, b) for a, b in zip(last_n_start_indices, last_time_start_indices)]
        lat = df["latitude"].to_numpy()
        lon = df["longitude"].to_numpy()

        recent_dists_and_times = pd.Series([
            # If there is at least one prior point in the window, compute the differences.
            np.array([
                haversine(lon[start_idx:i], lat[start_idx:i], lon[i], lat[i]) if i > start_idx else np.array([]),
                timestamps[i] - timestamps[start_idx:i] if i > start_idx else np.array([])
            ]) if start_idx < i else np.empty((2, 0))
            for i, start_idx in enumerate(start_indices)
        ], index=df.index)
        return recent_dists_and_times

    def get_last_trip_end_point_idx(self, curr_idx, recent_diffs: np.ndarray):
        """
        Determines the appropriate trip end point index based on recent differences.
        This function is adapted from the time filter segmentation logic.

        :param curr_idx: The current index where a potential trip end was detected.
        :param recent_diffs: A numpy array (shape (2, N)) containing recent distance and time differences.
        :return: A tuple (ended_before_this, last_trip_end_index) where ended_before_this is a boolean
                 indicating if the end of the trip occurred before the current index.
        """
        if recent_diffs.shape[1] == 0:
            return (False, curr_idx)

        # Remove any NaN values from the distance differences.
        recent_diffs_non_na = recent_diffs[:, ~np.isnan(recent_diffs[0, :])]
        num_recent_diffs_in_point_threshold = min(len(recent_diffs_non_na[0, :]), self.point_threshold)
        num_recent_diffs_in_time_threshold = np.sum(recent_diffs_non_na[1, :] < self.time_threshold)
        ended_before_this = (num_recent_diffs_in_time_threshold == 0)

        # Use the median index over the recent window as a heuristic for the trip end.
        last_n_median_idx = np.median(np.arange(curr_idx - num_recent_diffs_in_point_threshold, curr_idx + 1))
        if ended_before_this:
            last_trip_end_index = int(last_n_median_idx)
        else:
            last_time_median_idx = np.median(np.arange(curr_idx - num_recent_diffs_in_time_threshold, curr_idx))
            last_trip_end_index = int(min(last_time_median_idx, last_n_median_idx))

        logging.debug("curr_idx = %d, recent_diffs_in_point_threshold = %d, recent_diffs_in_time_threshold = %d, "
                      "ended_before_this = %s, last_trip_end_index = %d",
                      curr_idx, num_recent_diffs_in_point_threshold,
                      num_recent_diffs_in_time_threshold,
                      ended_before_this, last_trip_end_index)
        return (ended_before_this, last_trip_end_index)
