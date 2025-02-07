from __future__ import division, unicode_literals, print_function, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import *
from past.utils import old_div
import logging
import attrdict as ad
import numpy as np
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
        Determines segmentation points for points generated using a distance filter.
        For these points, a trip end is detected when the device has dwelled in place.
        """
        self.time_threshold = time_threshold
        self.point_threshold = point_threshold
        self.distance_threshold = distance_threshold
        # Precompute a constant speed threshold used to decide if the device is nearly still.
        # (This matches the original: (2*distance_threshold) / (time_threshold/2) == 4*distance_threshold/time_threshold)
        self.speed_threshold = (4.0 * distance_threshold) / time_threshold

    def segment_into_trips(self, timeseries, time_query, filtered_points_df):
        # Get the filtered points; preserve the original pipeline time names.
        with ect.Timer() as t_get_filtered_points:
            # Use a copy so that we can update columns without side effects.
            self.filtered_points_df = filtered_points_df.copy()
            user_id = self.filtered_points_df["user_id"].iloc[0]
            self.filtered_points_df["valid"] = True  # mark all points as initially valid.
        esds.store_pipeline_time(
            user_id,
            ecwp.PipelineStages.TRIP_SEGMENTATION.name + "/segment_into_trips_dist/get_filtered_points_df",
            time.time(),
            t_get_filtered_points.elapsed
        )

        self.transition_df = timeseries.get_data_df("statemachine/transition", time_query)
        self.motion_list = list(timeseries.find_entries(["background/motion_activity"], time_query))
        self.last_ts_processed = None

        segmentation_points = []
        curr_trip_start_point = None
        just_ended = True  # When True, we are waiting for a point that can start a new trip.

        with ect.Timer() as t_loop:
            # Iterate over rows using itertuples for improved performance.
            for idx, row in enumerate(self.filtered_points_df.itertuples(index=False)):
                # Convert the namedtuple to an AttrDict.
                currPoint = ad.AttrDict(row._asdict())
                currPoint.idx = idx
                # WORKAROUND: itertuples omits columns whose names start with an underscore.
                if '_id' not in currPoint:
                    currPoint['_id'] = self.filtered_points_df.iloc[idx]['_id']

                if just_ended:
                    # If the current point is very near the previous one, join it with the previous trip.
                    if self.continue_just_ended(idx, currPoint, self.filtered_points_df):
                        self.last_ts_processed = currPoint.metadata_write_ts
                        continue
                    else:
                        curr_trip_start_point = currPoint
                        just_ended = False
                        continue

                # Otherwise, check if the trip should end.
                lastPoint = self.find_last_valid_point(idx)
                if self.has_trip_ended(lastPoint, currPoint, timeseries):
                    segmentation_points.append((curr_trip_start_point, lastPoint))
                    self.last_ts_processed = currPoint.metadata_write_ts
                    just_ended = True
                    # Process the current point: if it should not be concatenated with the previous trip,
                    # treat it as the new trip start.
                    if not self.continue_just_ended(idx, currPoint, self.filtered_points_df):
                        curr_trip_start_point = currPoint
                        just_ended = False

            # If the final trip never had an explicit end but transitions indicate a stop, end the trip.
            if not just_ended and (len(self.transition_df) > 0):
                lastPoint = ad.AttrDict(self.filtered_points_df.iloc[-1])
                stopped_moving_after_last = self.transition_df[
                    (self.transition_df.ts > lastPoint.ts) & (self.transition_df.transition == 2)
                ]
                if not stopped_moving_after_last.empty:
                    segmentation_points.append((curr_trip_start_point, lastPoint))
                    self.last_ts_processed = lastPoint.metadata_write_ts

        esds.store_pipeline_time(
            user_id,
            ecwp.PipelineStages.TRIP_SEGMENTATION.name + "/segment_into_trips_dist/loop",
            time.time(),
            t_loop.elapsed
        )

        return segmentation_points

    def has_trip_ended(self, lastPoint, currPoint, timeseries):
        """
        Determines whether the gap between lastPoint and currPoint indicates that the trip has ended.
        The decision is based on elapsed time, distance traveled, computed speed, and whether tracking was restarted.
        """
        timeDelta = currPoint.ts - lastPoint.ts
        distDelta = pf.calDistance(lastPoint, currPoint)

        if timeDelta > self.time_threshold:
            speedDelta = (distDelta / timeDelta) if timeDelta > 0 else np.nan

            if eaisr.is_tracking_restarted_in_range(lastPoint.ts, currPoint.ts, timeseries, self.transition_df):
                logging.debug("Tracking restarted between %s and %s", lastPoint.ts, currPoint.ts)
                return True

            # Check if there is ongoing motion between the two points.
            ongoing_motion = eaisr.get_ongoing_motion_in_range(lastPoint.ts, currPoint.ts, timeseries, self.motion_list)
            if not ongoing_motion:
                logging.debug("No ongoing motion from %s to %s", lastPoint.ts, currPoint.ts)
                return True

            TWELVE_HOURS = 12 * 60 * 60
            if timeDelta > TWELVE_HOURS:
                return True

            # If the computed speed is below the threshold then we are likely still.
            if speedDelta < self.speed_threshold:
                # Check for a huge invalid timestamp offset; if so, drop this point.
                if eaistc.is_huge_invalid_ts_offset(self, lastPoint, currPoint, timeseries, ongoing_motion):
                    logging.debug("Dropping point %s due to huge invalid ts offset", currPoint.idx)
                    self.filtered_points_df.at[currPoint.idx, "valid"] = False
                    timeseries.invalidate_raw_entry(currPoint["_id"])
                    return False
                else:
                    return True

        return False

    def find_last_valid_point(self, idx):
        """
        Walk backwards from idx until a valid point is found. (Typically the immediately preceding point is valid.)
        """
        i = idx - 1
        while i >= 0:
            point = ad.AttrDict(self.filtered_points_df.iloc[i])
            if point.valid:
                return point
            i -= 1
        return ad.AttrDict(self.filtered_points_df.iloc[0])

    def continue_just_ended(self, idx, currPoint, filtered_points_df):
        """
        If the current point is very near the immediately preceding point (both in time and distance),
        then treat it as part of the just-ended trip rather than the start of a new trip.
        """
        if idx == 0:
            return False
        lastPoint = ad.AttrDict(filtered_points_df.iloc[idx - 1])
        deltaDist = pf.calDistance(lastPoint, currPoint)
        deltaTime = currPoint.ts - lastPoint.ts
        logging.debug("continue_just_ended: deltaDist=%s (threshold=%s), deltaTime=%s (threshold=%s)",
                      deltaDist, self.distance_threshold, deltaTime, self.time_threshold)
        return deltaDist < self.distance_threshold
