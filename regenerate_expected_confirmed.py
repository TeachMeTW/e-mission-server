#!/usr/bin/env python
# Script to regenerate expected_confirmed_trips and expected_confirmed_places files

import json
import logging
import argparse
import numpy as np
import os

import emission.tests.common as etc
import emission.storage.json_wrappers as esj
import emission.core.wrapper.localdate as ecwl
import emission.storage.timeseries.abstract_timeseries as esta
import emission.core.get_database as edb
import emission.core.wrapper.entry as ecwe
import emission.core.wrapper.user as ecwu

def regenerate_expected_confirmed(dataFile, date_str, preload=True, 
                                 trip_user_inputs=None, place_user_inputs=None):
    """
    Regenerate expected confirmed trips and places for a given data file
    """
    logging.info(f"Regenerating expected confirmed data for {dataFile} with date {date_str}")
    
    # Parse the date
    date_parts = date_str.split('-')
    local_date = ecwl.LocalDate({'year': int(date_parts[0]), 'month': int(date_parts[1]), 'day': int(date_parts[2])})
    
    # Set up test environment
    test_case = type('obj', (object,), {
        'testUUID': None,  # Will be set by setupRealExample
        'evaluation': False,
    })
    
    # Set up test with real example data
    etc.setupRealExample(test_case, dataFile)
    user_id = test_case.testUUID
    
    # Create suffixes based on user inputs
    ct_suffix = "".join(".manual_" + k for k in trip_user_inputs) if trip_user_inputs else ""
    cp_suffix = "".join(".manual_" + k for k in place_user_inputs) if place_user_inputs else ""
    
    # Handle user inputs based on preload setting
    if preload:
        if trip_user_inputs or place_user_inputs:
            input_file = f"{dataFile}.user_inputs{ct_suffix}{cp_suffix}"
        else:
            input_file = f"{dataFile}.user_inputs"
            
        if os.path.exists(input_file):
            entries = json.load(open(input_file), object_hook=esj.wrapped_object_hook)
            # Set up real example with entries
            test_case.entries = entries
            etc.setupRealExampleWithEntries(test_case)
        
    # Run the intake pipeline to process the data
    etc.runIntakePipeline(user_id)
    
    # If not preload, load user inputs after first pipeline run
    if not preload:
        if trip_user_inputs or place_user_inputs:
            input_file = f"{dataFile}.user_inputs{ct_suffix}{cp_suffix}"
        else:
            input_file = f"{dataFile}.user_inputs"
            
        if os.path.exists(input_file):
            entries = json.load(open(input_file), object_hook=esj.wrapped_object_hook)
            # Set up real example with entries
            test_case.entries = entries
            etc.setupRealExampleWithEntries(test_case)
            # Run the intake pipeline again
            etc.runIntakePipeline(user_id)
    
    # Get the time series for this user
    ts = esta.TimeSeries.get_time_series(user_id)
    
    # Get confirmed trips and places
    confirmed_trips = ts.find_entries(["analysis/confirmed_trip"], None)
    confirmed_places = ts.find_entries(["analysis/confirmed_place"], None)
    
    # Save confirmed trips
    output_trips_filename = f"{dataFile}.expected_confirmed_trips{ct_suffix}"
    with open(output_trips_filename, "w") as outfile:
        json.dump(confirmed_trips, outfile, indent=4, default=esj.wrapped_default)
    logging.info(f"Saved {len(confirmed_trips)} confirmed trips to {output_trips_filename}")
    
    # Save confirmed places
    output_places_filename = f"{dataFile}.expected_confirmed_places{cp_suffix}"
    with open(output_places_filename, "w") as outfile:
        json.dump(confirmed_places, outfile, indent=4, default=esj.wrapped_default)
    logging.info(f"Saved {len(confirmed_places)} confirmed places to {output_places_filename}")
    
    # Clear the test data
    if os.environ.get("SKIP_TEARDOWN", False):
        logging.info("SKIP_TEARDOWN = true, not clearing related databases")
        ecwu.User.registerWithUUID("automated_tests", user_id)
    else:
        clear_user_data(user_id)
        logging.info("Cleared test data")

def clear_user_data(user_id):
    """Clear all data for this test user"""
    logging.info(f"Clearing data for user {user_id}")
    edb.get_timeseries_db().delete_many({"user_id": user_id})
    edb.get_analysis_timeseries_db().delete_many({"user_id": user_id})
    edb.get_usercache_db().delete_many({"user_id": user_id})
    edb.get_pipeline_state_db().delete_many({"user_id": user_id})

def main():
    parser = argparse.ArgumentParser(description="Regenerate expected_confirmed_trips and expected_confirmed_places files")
    parser.add_argument('--datafile', type=str, required=True,
                        help="Path to the data file (e.g., emission/tests/data/real_examples/shankari_2016-06-20)")
    parser.add_argument('--date', type=str, required=True,
                        help="Date in YYYY-MM-DD format (e.g., 2016-06-20)")
    parser.add_argument('--postload', action='store_true',
                        help="Load user inputs after the pipeline is run (default is preload)")
    parser.add_argument('--trip-inputs', type=str, nargs='*',
                        help="Trip user input types (e.g., trip_user_input)")
    parser.add_argument('--place-inputs', type=str, nargs='*',
                        help="Place user input types (e.g., place_addition_input)")
    parser.add_argument('--debug', action='store_true',
                        help="Print debug information")
    args = parser.parse_args()
    
    # Set up logging
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Set random seed for reproducibility
    np.random.seed(61297777)
    
    # Regenerate expected confirmed data
    regenerate_expected_confirmed(
        args.datafile,
        args.date,
        preload=(not args.postload),
        trip_user_inputs=args.trip_inputs,
        place_user_inputs=args.place_inputs
    )
    
    logging.info("Expected confirmed data regeneration complete")

if __name__ == '__main__':
    main() 