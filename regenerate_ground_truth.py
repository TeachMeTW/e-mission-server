#!/usr/bin/env python
# Script to regenerate ground truth files with the sensed modes from rule_engine.py

import json
import logging
import uuid
import argparse
import copy
from pathlib import Path

import emission.tests.common as etc
import emission.analysis.plotting.geojson.geojson_feature_converter as gfc
import emission.core.wrapper.localdate as ecwl
import emission.storage.json_wrappers as esj
import emission.analysis.classification.inference.mode.rule_engine as eacimr
import emission.storage.timeseries.abstract_timeseries as esta
import emission.storage.decorations.analysis_timeseries_queries as esda
import emission.core.wrapper.entry as ecwe
import emission.core.wrapper.modeprediction as ecwm
import emission.analysis.section_features as easf
import emission.core.get_database as edb

import time
import os
import numpy as np

def regenerate_ground_truth(dataFile, date_str, output_suffix=""):
    logging.info(f"Regenerating ground truth for {dataFile} with date {date_str}")
    
    # Set up test environment
    test_case = type('obj', (object,), {
        'testUUID': None,  # Will be set by setupRealExample
    })
    etc.setupRealExample(test_case, dataFile)
    user_id = test_case.testUUID
    
    # Run the intake pipeline to process the data
    etc.runIntakePipeline(user_id)
    
    # Parse the date
    date_parts = date_str.split('-')
    local_date = ecwl.LocalDate({'year': int(date_parts[0]), 'month': int(date_parts[1]), 'day': int(date_parts[2])})
    
    # Run the rule engine to update the sensed modes
    update_sensed_modes_with_rule_engine(user_id)
    
    # Get the geojson data for the specified date
    api_result = gfc.get_geojson_for_dt(user_id, local_date, local_date)
    
    # Create the output filename
    output_filename = f"{dataFile}.ground_truth{output_suffix}"
    
    # Save the ground truth
    with open(output_filename, "w") as outfile:
        wrapped_gt = {
            "data": api_result,
            "metadata": {
                "key": f"diary/trips-{date_str}",
                "type": "document",
                "write_ts": int(time.time())
            }
        }
        json.dump(wrapped_gt, outfile, indent=4, default=esj.wrapped_default)
    
    logging.info(f"Saved ground truth to {output_filename}")
    
    # Clear the test data
    clear_user_data(user_id)

def update_sensed_modes_with_rule_engine(user_id):
    """
    Update the sensed modes in all inferred sections using the rule engine
    """
    logging.info(f"Updating sensed modes for user {user_id} using rule engine")
    
    # Get the time series for this user
    ts = esta.TimeSeries.get_time_series(user_id)
    
    # Get all cleaned sections for this user
    # Use None for time_query to get all sections
    cleaned_sections = esda.get_entries(esda.CLEANED_SECTION_KEY, user_id, time_query=None)
    
    if len(cleaned_sections) == 0:
        logging.warning("No cleaned sections found for user")
        return
    
    # Process each section with the rule engine
    for i, section_entry in enumerate(cleaned_sections):
        section_id = section_entry.get_id()
        logging.debug(f"Processing section {i+1}/{len(cleaned_sections)}: {section_id}")
        
        # Get the prediction from the rule engine
        try:
            if hasattr(section_entry.data, 'sensed_mode') and section_entry.data.sensed_mode == "AIR_OR_HSR":
                predicted_prob = {'AIR_OR_HSR': 1}
            else:
                predicted_prob = eacimr.get_prediction(i, section_entry)
        except Exception as e:
            logging.error(f"Error getting prediction for section {section_id}: {e}")
            predicted_prob = {'UNKNOWN': 1}
        
        try:
            # Create and insert the mode prediction
            mp = ecwm.Modeprediction()
            mp.trip_id = section_entry.data.trip_id
            mp.section_id = section_id
            mp.start_ts = section_entry.data.start_ts
            mp.end_ts = section_entry.data.end_ts
            mp.algorithm_id = ecwm.AlgorithmTypes.SIMPLE_RULE_ENGINE
            mp.predicted_mode_map = predicted_prob
            ts.insert_data(user_id, "inference/prediction", mp)
            
            # Create the inferred section with the updated sensed mode
            is_dict = copy.deepcopy(section_entry)
            if "_id" in is_dict:
                del is_dict["_id"]
                
            # Make sure metadata exists
            if "metadata" not in is_dict:
                is_dict["metadata"] = {}
                
            is_dict["metadata"]["key"] = "analysis/inferred_section"
            
            # Make sure data exists
            if "data" not in is_dict:
                is_dict["data"] = {}
                
            is_dict["data"]["sensed_mode"] = ecwm.PredictedModeTypes[easf.select_inferred_mode([mp])].value
            is_dict["data"]["cleaned_section"] = section_id
            
            # Insert the new inferred section
            ise = ecwe.Entry(is_dict)
            logging.debug(f"Updated sensed mode for section {section_id} to {ise.data.sensed_mode}")
            ts.insert(ise)
        except Exception as e:
            logging.error(f"Error processing section {section_id}: {e}")
    
    logging.info(f"Updated {len(cleaned_sections)} sections with rule engine predictions")

def clear_user_data(user_id):
    """Clear all data for this test user"""
    logging.info(f"Clearing data for user {user_id}")
    edb.get_timeseries_db().delete_many({"user_id": user_id})
    edb.get_analysis_timeseries_db().delete_many({"user_id": user_id})
    edb.get_usercache_db().delete_many({"user_id": user_id})
    edb.get_pipeline_state_db().delete_many({"user_id": user_id})

def main():
    parser = argparse.ArgumentParser(description="Regenerate ground truth files with sensed modes from rule_engine.py")
    parser.add_argument('--first', action='store_true', 
                        help="Regenerate the first file (shankari_2016-07-22)")
    parser.add_argument('--second', action='store_true',
                        help="Regenerate the second file (shankari_2016-07-25)")
    parser.add_argument('--both', action='store_true',
                        help="Regenerate both files")
    parser.add_argument('--debug', action='store_true',
                        help="Print debug information")
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Set random seed for reproducibility
    np.random.seed(61297777)
    
    # Determine which files to regenerate
    regenerate_first = args.first or args.both
    regenerate_second = args.second or args.both
    
    if not (regenerate_first or regenerate_second):
        logging.warning("No files specified for regeneration. Please use --first, --second, or --both.")
        return
    
    # Define file paths
    first_file = "emission/tests/data/real_examples/shankari_2016-07-22"
    second_file = "emission/tests/data/real_examples/shankari_2016-07-25"
    
    # Regenerate the first file
    if regenerate_first:
        regenerate_ground_truth(first_file, "2016-07-22")
        if args.debug:
            with open(first_file + ".ground_truth") as f:
                data = json.load(f)
                print(f"First file has {len(data['data'])} trips")
                for i, trip in enumerate(data['data']):
                    print(f"Trip {i+1}: {trip['properties']['start_fmt_time']} -> {trip['properties']['end_fmt_time']}")
    
    # Regenerate the second file
    if regenerate_second:
        regenerate_ground_truth(second_file, "2016-07-25")
        if args.debug:
            with open(second_file + ".ground_truth") as f:
                data = json.load(f)
                print(f"Second file has {len(data['data'])} trips")
                for i, trip in enumerate(data['data']):
                    print(f"Trip {i+1}: {trip['properties']['start_fmt_time']} -> {trip['properties']['end_fmt_time']}")
    
    logging.info("Ground truth regeneration complete")

if __name__ == '__main__':
    main() 