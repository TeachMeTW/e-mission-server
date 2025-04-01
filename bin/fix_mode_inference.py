#!/usr/bin/env python
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import *

import logging
import sys
import argparse
import uuid
import arrow
import copy

import emission.core.get_database as edb
import emission.storage.timeseries.abstract_timeseries as esta
import emission.storage.decorations.analysis_timeseries_queries as esda
import emission.storage.timeseries.timequery as estt
import emission.storage.pipeline_queries as epq

import emission.core.wrapper.entry as ecwe
import emission.core.wrapper.modeprediction as ecwm
import emission.core.wrapper.motionactivity as ecwma

import emission.analysis.section_features as easf
import emission.analysis.classification.inference.mode.rule_engine as eacimr
import emission.analysis.userinput.matcher as eaum
import emission.analysis.plotting.composite_trip_creation as eapc

"""
Script to fix mode inference issues without rerunning the entire pipeline.
Only processes sections where mode inference previously failed or had issues.

Usage:
  python bin/fix_mode_inference.py <user_id>
  python bin/fix_mode_inference.py <user_id> --start_date YYYY-MM-DD --end_date YYYY-MM-DD
  python bin/fix_mode_inference.py --all_users
"""

def parse_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--user_id", help="User ID to process")
    group.add_argument("--all_users", action="store_true", help="Process all users")
    parser.add_argument("--start_date", help="Start date (YYYY-MM-DD) - if not specified, will process all data")
    parser.add_argument("--end_date", help="End date (YYYY-MM-DD) - if not specified, will process until latest data")
    parser.add_argument("--verbose", "-v", action="store_true", help="Turn on verbose logging")
    parser.add_argument("--dry_run", action="store_true", help="Dry run without modifying the database")
    return parser.parse_args()

def get_time_query(args):
    time_query = None
    if args.start_date is not None or args.end_date is not None:
        time_query = estt.TimeQuery("data.start_ts", "data.end_ts")
        if args.start_date is not None:
            time_query.startTs = arrow.get(args.start_date).float_timestamp
        if args.end_date is not None:
            time_query.endTs = arrow.get(args.end_date).shift(days=1).float_timestamp  # Include the entire end day
    return time_query

def get_users_to_process(args):
    if args.all_users:
        return [e["user_id"] for e in edb.get_pipeline_state_db().find({})]
    else:
        return [uuid.UUID(args.user_id)]

def fix_mode_inference_for_user(user_id, time_query, dry_run=False):
    """
    Run mode inference only for sections that don't have corresponding inferred sections
    or where the inferred mode might have been incorrect.
    """
    logging.info(f"Processing user {user_id}")
    ts = esta.TimeSeries.get_time_series(user_id)
    
    # Get all cleaned sections for this user in the time range
    cleaned_sections = esda.get_entries(esda.CLEANED_SECTION_KEY, user_id, time_query=time_query)
    
    if len(cleaned_sections) == 0:
        logging.warning(f"No cleaned sections found for user {user_id} in the specified time range")
        return
    
    logging.info(f"Found {len(cleaned_sections)} cleaned sections for user {user_id}")
    
    # Group sections by their trip ID
    trip_to_sections = {}
    for section in cleaned_sections:
        trip_id = section.data.trip_id
        if trip_id not in trip_to_sections:
            trip_to_sections[trip_id] = []
        trip_to_sections[trip_id].append(section)
    
    # Process each section
    sections_fixed = 0
    sections_processed = 0
    
    for trip_id, sections in trip_to_sections.items():
        logging.debug(f"Processing trip {trip_id} with {len(sections)} sections")
        
        # Sort sections by start time for consistent processing
        sections.sort(key=lambda x: x.data.start_ts)
        
        # Track which sections need their mode inferred
        sections_to_fix = []
        
        for section in sections:
            section_id = section.get_id()
            
            # Check if this section already has a valid inferred section
            inferred_sections = list(ts.find_entries(
                ["analysis/inferred_section"],
                estt.TimeQuery("data.start_ts", "data.end_ts",
                              startTs=section.data.start_ts,
                              endTs=section.data.end_ts)
            ))
            
            # Find matching inferred section by section_id reference
            matching_inferred = None
            for inf_section in inferred_sections:
                if 'cleaned_section' in inf_section.data and inf_section.data.cleaned_section == section_id:
                    matching_inferred = inf_section
                    break
            
            # Decide if we need to fix this section
            needs_fixing = False
            
            if matching_inferred is None:
                logging.info(f"Section {section_id} needs fixing: No matching inferred section found")
                needs_fixing = True
            else:
                # Check if the inferred mode is UNKNOWN, which suggests there was an error
                if matching_inferred.data.sensed_mode == ecwm.PredictedModeTypes.UNKNOWN.value:
                    logging.info(f"Section {section_id} needs fixing: Inferred mode is UNKNOWN")
                    needs_fixing = True
                    
                    # Delete the existing inferred section so we can recreate it
                    if not dry_run:
                        ts.delete_entry(matching_inferred.get_id())
                        
                        # Also delete any associated predictions
                        predictions = list(ts.find_entries(
                            ["inference/prediction"],
                            estt.TimeQuery("data.start_ts", "data.end_ts",
                                         startTs=section.data.start_ts,
                                         endTs=section.data.end_ts)
                        ))
                        
                        for pred in predictions:
                            if pred.data.section_id == section_id:
                                ts.delete_entry(pred.get_id())
            
            if needs_fixing:
                sections_to_fix.append(section)
        
        # If there are sections to fix in this trip, process them
        if sections_to_fix:
            for i, section in enumerate(sections_to_fix):
                section_id = section.get_id()
                logging.info(f"Fixing section {section_id}")
                
                if dry_run:
                    logging.info(f"[DRY RUN] Would fix mode inference for section {section_id}")
                    sections_processed += 1
                    continue
                
                try:
                    # Get the prediction from the rule engine
                    if section.data.sensed_mode == ecwma.MotionTypes.AIR_OR_HSR.value:
                        predicted_prob = {'AIR_OR_HSR': 1}
                    else:
                        predicted_prob = eacimr.get_prediction(i, section)
                    
                    # Create and insert the mode prediction
                    mp = ecwm.Modeprediction()
                    mp.trip_id = section.data.trip_id
                    mp.section_id = section_id
                    mp.algorithm_id = ecwm.AlgorithmTypes.SIMPLE_RULE_ENGINE
                    mp.predicted_mode_map = predicted_prob
                    mp.start_ts = section.data.start_ts
                    mp.end_ts = section.data.end_ts
                    ts.insert_data(user_id, "inference/prediction", mp)
                    
                    # Create the inferred section
                    is_dict = copy.deepcopy(section)
                    if "_id" in is_dict:
                        del is_dict["_id"]
                    
                    is_dict["metadata"]["key"] = "analysis/inferred_section"
                    is_dict["data"]["sensed_mode"] = ecwm.PredictedModeTypes[easf.select_inferred_mode([mp])].value
                    is_dict["data"]["cleaned_section"] = section_id
                    
                    # Insert the new inferred section
                    ise = ecwe.Entry(is_dict)
                    logging.debug(f"Updated sensed mode for section {section_id} to {ise.data.sensed_mode}")
                    ts.insert(ise)
                    
                    sections_fixed += 1
                except Exception as e:
                    logging.error(f"Error processing section {section_id}: {e}")
                    
                sections_processed += 1
            
            # Now update any confirmed trips and composite trips that reference this trip
            if not dry_run:
                update_trip_summaries(user_id, trip_id)
    
    logging.info(f"User {user_id}: Processed {sections_processed} sections, fixed {sections_fixed} sections")
    return sections_fixed

def update_trip_summaries(user_id, trip_id):
    """
    Update the inferred_section_summary field in confirmed trips and composite trips
    that reference the given trip_id.
    """
    ts = esta.TimeSeries.get_time_series(user_id)
    
    # Find the cleaned trip with this trip_id
    cleaned_trip = None
    cleaned_trips = list(ts.find_entries(
        [esda.CLEANED_TRIP_KEY],
        estt.TimeQuery("data.start_ts", "data.end_ts")
    ))
    
    for ct in cleaned_trips:
        if ct.get_id() == trip_id:
            cleaned_trip = ct
            break
    
    if cleaned_trip is None:
        logging.warning(f"Could not find cleaned trip with ID {trip_id}")
        return
    
    # Find confirmed trips that reference this cleaned trip
    confirmed_trips = list(ts.find_entries(
        [esda.CONFIRMED_TRIP_KEY],
        estt.TimeQuery("data.start_ts", "data.end_ts")
    ))
    
    confirmed_to_update = []
    for ct in confirmed_trips:
        if 'cleaned_trip' in ct.data and ct.data.cleaned_trip == trip_id:
            confirmed_to_update.append(ct)
    
    # Update confirmed trips
    for ct in confirmed_to_update:
        logging.info(f"Updating confirmed trip {ct.get_id()}")
        
        # Generate updated summaries
        inferred_section_summary = eaum.get_section_summary(ts, cleaned_trip, "analysis/inferred_section")
        
        # Update the confirmed trip
        update_dict = {
            "data.inferred_section_summary": inferred_section_summary
        }
        
        edb.get_analysis_timeseries_db().update_one(
            {"_id": ct.get_id()},
            {"$set": update_dict}
        )
        
        # Find and update the corresponding composite trip
        composite_trips = list(ts.find_entries(
            [esda.COMPOSITE_TRIP_KEY],
            estt.TimeQuery("data.start_ts", "data.end_ts")
        ))
        
        for comp_trip in composite_trips:
            if 'confirmed_trip' in comp_trip.data and comp_trip.data.confirmed_trip == ct.get_id():
                logging.info(f"Updating composite trip {comp_trip.get_id()}")
                
                # Update the composite trip
                comp_update_dict = {
                    "data.inferred_section_summary": inferred_section_summary,
                    "data.sections": eapc.get_sections_for_confirmed_trip(ct)
                }
                
                edb.get_analysis_timeseries_db().update_one(
                    {"_id": comp_trip.get_id()},
                    {"$set": comp_update_dict}
                )

def main():
    args = parse_args()
    
    # Set up logging
    logging_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                      level=logging_level, stream=sys.stdout)
    
    # Get time query based on specified date range
    time_query = get_time_query(args)
    
    # Get list of users to process
    users = get_users_to_process(args)
    
    total_users = len(users)
    logging.info(f"Processing {total_users} users")
    
    total_fixed = 0
    for i, user_id in enumerate(users):
        logging.info(f"Processing user {i+1}/{total_users}: {user_id}")
        sections_fixed = fix_mode_inference_for_user(user_id, time_query, args.dry_run)
        total_fixed += sections_fixed if sections_fixed else 0
    
    logging.info(f"Fix mode inference complete! Fixed {total_fixed} sections across {total_users} users")

if __name__ == '__main__':
    main() 