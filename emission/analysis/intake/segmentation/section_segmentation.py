from __future__ import print_function, unicode_literals, division, absolute_import
# Standard imports
from future import standard_library
standard_library.install_aliases()
from builtins import *
from builtins import object
import logging
import bisect
import numpy as np

# Our imports
import emission.analysis.configs.dynamic_config as eadc
import emission.storage.pipeline_queries as epq
import emission.storage.decorations.analysis_timeseries_queries as esda
import emission.storage.timeseries.abstract_timeseries as esta

import emission.core.wrapper.motionactivity as ecwm
import emission.core.wrapper.location as ecwl
import emission.core.wrapper.section as ecwc
import emission.core.wrapper.stop as ecws
import emission.core.wrapper.entry as ecwe

import emission.core.common as ecc
import emcommon.bluetooth.ble_matching as emcble


class SectionSegmentationMethod(object):
    def segment_into_sections(self, timeseries, distance_from_place, time_query):
        """
        Examines the timeseries database for a specific range and returns the
        points at which the trip needs to be segmented.
        Returns an array of tuples: [(start1, end1), (start2, end2), ...].
        If there are no segments, returns an empty array.
        TODO: Consider handling cases where segmentation is not ready.
        """
        pass


def segment_current_sections(user_id):
    time_query = epq.get_time_range_for_sectioning(user_id)
    try:
        trips_to_process = esda.get_entries(esda.RAW_TRIP_KEY, user_id, time_query)
        for trip_entry in trips_to_process:
            logging.info("+" * 20 + ("Processing trip %s for user %s" %
                                     (trip_entry.get_id(), user_id)) + "+" * 20)
            segment_trip_into_sections(user_id, trip_entry, trip_entry.data.source)
        last_trip_processed = trips_to_process[-1] if trips_to_process else None
        epq.mark_sectioning_done(user_id, last_trip_processed)
    except Exception:
        logging.exception("Sectioning failed for user %s" % user_id)
        epq.mark_sectioning_failed(user_id)


def segment_trip_into_sections(user_id, trip_entry, trip_source):
    ts = esta.TimeSeries.get_time_series(user_id)
    time_query = esda.get_time_query_for_trip_like(esda.RAW_TRIP_KEY, trip_entry.get_id())
    distance_from_place = _get_distance_from_start_place_to_end(trip_entry)
    
    # Preload filtered location entries for the time range.
    filtered_loc_entries = list(ts.find_entries(["background/filtered_location"], time_query))
    
    # If entries are not guaranteed to be sorted, uncomment the next line:
    # filtered_loc_entries.sort(key=lambda entry: entry["data"]["ts"])
    
    # Build a NumPy array of timestamps for fast lookup.
    timestamps = np.array([entry["data"]["ts"] for entry in filtered_loc_entries])
    
    # Preload BLE entries during the trip.
    ble_entries_during_trip = list(ts.find_entries(["background/bluetooth_ble"], time_query))
    
    # Simple cache to avoid repeated Location conversions.
    location_cache = {}

    def get_entry_at_ts(target_ts):
        # Use np.searchsorted to find the insertion point (using 'right' gives the floor index).
        idx = np.searchsorted(timestamps, target_ts, side='right') - 1
        if idx < 0:
            idx = 0
        return filtered_loc_entries[idx]

    def get_loc_for_ts(timestamp):
        # Return cached location if available.
        if timestamp in location_cache:
            return location_cache[timestamp]
        entry = get_entry_at_ts(timestamp)
        loc = ecwl.Location(entry["data"])
        location_cache[timestamp] = loc
        return loc

    # Choose segmentation method based on trip_source.
    if trip_source == "DwellSegmentationTimeFilter":
        import emission.analysis.intake.segmentation.section_segmentation_methods.smoothed_high_confidence_motion as shcm
        segmentation_method = shcm.SmoothedHighConfidenceMotion(
            60, 100,
            [ecwm.MotionTypes.TILTING,
             ecwm.MotionTypes.UNKNOWN,
             ecwm.MotionTypes.STILL,
             ecwm.MotionTypes.NONE,
             ecwm.MotionTypes.STOPPED_WHILE_IN_VEHICLE]
        )
    elif trip_source == "DwellSegmentationDistFilter":
        import emission.analysis.intake.segmentation.section_segmentation_methods.smoothed_high_confidence_with_visit_transitions as shcmvt
        segmentation_method = shcmvt.SmoothedHighConfidenceMotionWithVisitTransitions(
            49, 50,
            [ecwm.MotionTypes.TILTING,
             ecwm.MotionTypes.UNKNOWN,
             ecwm.MotionTypes.STILL,
             ecwm.MotionTypes.NONE,  # iOS only
             ecwm.MotionTypes.STOPPED_WHILE_IN_VEHICLE]  # iOS only
        )
    else:
        raise ValueError("Unknown trip source: {}".format(trip_source))
    
    # Compute segmentation points.
    segmentation_points = segmentation_method.segment_into_sections(ts, distance_from_place, time_query)
    
    # Retrieve trip boundaries using our optimized lookup.
    trip_start_loc = get_loc_for_ts(trip_entry.data.start_ts)
    trip_end_loc = get_loc_for_ts(trip_entry.data.end_ts)
    logging.debug("trip_start_loc = %s, trip_end_loc = %s" % (trip_start_loc, trip_end_loc))
    
    # Cache the dynamic configuration once.
    dynamic_config = eadc.get_dynamic_config()
    
    # Helper to convert a segmentation row into location data.
    def get_loc_for_row(row):
        # We assume that ts.df_row_to_entry is efficient.
        return ts.df_row_to_entry("background/filtered_location", row).data

    prev_section_entry = None

    for i, (start_loc_doc, end_loc_doc, sensed_mode) in enumerate(segmentation_points):
        start_loc_data = get_loc_for_row(start_loc_doc)
        end_loc_data = get_loc_for_row(end_loc_doc)
        # Optionally cache these conversions too if needed.
        start_loc = ecwl.Location(start_loc_data)
        end_loc = ecwl.Location(end_loc_data)
        # For the first and last segments, force the trip boundaries.
        if prev_section_entry is None:
            start_loc = trip_start_loc
        if i == len(segmentation_points) - 1:
            end_loc = trip_end_loc
        
        # Determine BLE-sensed mode using preloaded BLE entries.
        ble_sensed_mode = emcble.get_ble_sensed_vehicle_for_section(
            ble_entries_during_trip, start_loc.ts, end_loc.ts, dynamic_config
        )
        
        section = ecwc.Section()
        section.trip_id = trip_entry.get_id()
        fill_section(section, start_loc, end_loc, sensed_mode, ble_sensed_mode)
        section_entry = ecwe.Entry.create_entry(user_id, esda.RAW_SECTION_KEY, section, create_id=True)
        
        if prev_section_entry is not None:
            stop = ecws.Stop()
            stop.trip_id = trip_entry.get_id()
            stop_entry = ecwe.Entry.create_entry(user_id, esda.RAW_STOP_KEY, stop, create_id=True)
            logging.debug("stop = %s, stop_entry = %s" % (stop, stop_entry))
            stitch_together(prev_section_entry, stop_entry, section_entry)
            ts.insert(stop_entry)
            ts.update(prev_section_entry)
        
        ts.insert(section_entry)
        prev_section_entry = section_entry


def fill_section(section, start_loc, end_loc, sensed_mode, ble_sensed_mode=None):
    section.start_ts = start_loc.ts
    section.start_local_dt = start_loc.local_dt
    section.start_fmt_time = start_loc.fmt_time

    section.end_ts = end_loc.ts
    try:
        section.end_local_dt = end_loc.local_dt
    except AttributeError:
        logging.error("Missing local_dt for location: %s" % end_loc)
    section.end_fmt_time = end_loc.fmt_time

    section.start_loc = start_loc.loc
    section.end_loc = end_loc.loc

    section.duration = end_loc.ts - start_loc.ts
    section.source = "SmoothedHighConfidenceMotion"
    section.sensed_mode = sensed_mode
    section.ble_sensed_mode = ble_sensed_mode


def stitch_together(ending_section_entry, stop_entry, starting_section_entry):
    ending_section = ending_section_entry.data
    stop = stop_entry.data
    starting_section = starting_section_entry.data

    ending_section.end_stop = stop_entry.get_id()

    stop.enter_ts = ending_section.end_ts
    stop.enter_local_dt = ending_section.end_local_dt
    stop.enter_fmt_time = ending_section.end_fmt_time
    stop.ending_section = ending_section_entry.get_id()

    stop.enter_loc = ending_section.end_loc
    stop.exit_loc = starting_section.start_loc
    stop.duration = starting_section.start_ts - ending_section.end_ts
    stop.distance = ecc.calDistance(stop.enter_loc.coordinates, stop.exit_loc.coordinates)
    stop.source = "SmoothedHighConfidenceMotion"

    stop.exit_ts = starting_section.start_ts
    stop.exit_local_dt = starting_section.start_local_dt
    stop.exit_fmt_time = starting_section.start_fmt_time
    stop.starting_section = starting_section_entry.get_id()

    starting_section.start_stop = stop_entry.get_id()

    ending_section_entry["data"] = ending_section
    stop_entry["data"] = stop
    starting_section_entry["data"] = starting_section


def _get_distance_from_start_place_to_end(raw_trip_entry):
    import emission.core.common as ecc
    start_place_id = raw_trip_entry.data.start_place
    start_place = esda.get_object(esda.RAW_PLACE_KEY, start_place_id)
    dist = ecc.calDistance(start_place.location.coordinates, raw_trip_entry.data.end_loc.coordinates)
    logging.debug("Distance from raw_place %s to the end of raw_trip_entry %s = %s" %
                  (start_place_id, raw_trip_entry.get_id(), dist))
    return dist
