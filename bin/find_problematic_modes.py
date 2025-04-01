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
import json
from collections import defaultdict
import pandas as pd

import emission.core.get_database as edb
import emission.storage.timeseries.abstract_timeseries as esta
import emission.storage.decorations.analysis_timeseries_queries as esda
import emission.storage.timeseries.timequery as estt

import emission.core.wrapper.modeprediction as ecwm
import emission.core.wrapper.motionactivity as ecwma

"""
Script to find users with unknown or problematic mode inferences.
Identifies:
1. Users with sections having UNKNOWN inferred modes
2. Users with cleaned sections that have no corresponding inferred sections

Usage:
  python bin/find_problematic_modes.py 
  python bin/find_problematic_modes.py --start_date YYYY-MM-DD --end_date YYYY-MM-DD
  python bin/find_problematic_modes.py --output_json
  python bin/find_problematic_modes.py --user_id UUID
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_id", help="Check only specific user ID")
    parser.add_argument("--start_date", help="Start date (YYYY-MM-DD) - if not specified, will process all data")
    parser.add_argument("--end_date", help="End date (YYYY-MM-DD) - if not specified, will process until latest data")
    parser.add_argument("--verbose", "-v", action="store_true", help="Turn on verbose logging")
    parser.add_argument("--output_json", action="store_true", help="Output results as JSON")
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

def get_users_to_check(args):
    if args.user_id:
        return [uuid.UUID(args.user_id)]
    else:
        # Get all users who have any data
        users = set()
        for key in [esda.CLEANED_SECTION_KEY, "analysis/inferred_section"]:
            collection = edb.get_timeseries_db().find_one({"metadata.key": key})
            if collection:
                user_ids = edb.get_timeseries_db().distinct("user_id", {"metadata.key": key})
                for user_id in user_ids:
                    users.add(user_id)
        return list(users)

def find_problematic_modes(user_id, time_query):
    """
    Find sections with problematic mode inferences for a user
    """
    result = {
        "unknown_modes": [],
        "missing_inferred_sections": [],
        "prefixed_modes": []
    }
    
    ts = esta.TimeSeries.get_time_series(user_id)
    
    # Get all cleaned sections
    cleaned_sections = esda.get_entries(esda.CLEANED_SECTION_KEY, user_id, time_query=time_query)
    if len(cleaned_sections) == 0:
        return None  # No sections for this user in the specified time range
    
    # Get all inferred sections
    inferred_sections = list(ts.find_entries(
        ["analysis/inferred_section"],
        time_query
    ))
    
    # Build a lookup of inferred sections by cleaned section id
    inferred_by_cleaned = {}
    for section in inferred_sections:
        if 'cleaned_section' in section.data:
            inferred_by_cleaned[section.data.cleaned_section] = section
    
    # Check each cleaned section
    for section in cleaned_sections:
        section_id = section.get_id()
        
        # Case 1: No matching inferred section
        if section_id not in inferred_by_cleaned:
            result["missing_inferred_sections"].append({
                "section_id": str(section_id),
                "start_time": arrow.get(section.data.start_ts).format("YYYY-MM-DD HH:mm:ss"),
                "duration": section.data.duration,
                "distance": section.data.distance
            })
            continue
            
        # Case 2: Section has UNKNOWN mode
        inferred = inferred_by_cleaned[section_id]
        if inferred.data.sensed_mode == ecwm.PredictedModeTypes.UNKNOWN.value:
            result["unknown_modes"].append({
                "section_id": str(section_id),
                "start_time": arrow.get(section.data.start_ts).format("YYYY-MM-DD HH:mm:ss"),
                "duration": section.data.duration,
                "distance": section.data.distance
            })
    
    # Check for prefixed modes in inference predictions
    predictions = list(ts.find_entries(
        ["inference/prediction"],
        time_query
    ))
    
    for pred in predictions:
        for mode, confidence in pred.data.predicted_mode_map.items():
            if ":" in mode:
                result["prefixed_modes"].append({
                    "section_id": str(pred.data.section_id),
                    "prefixed_mode": mode,
                    "confidence": confidence
                })
    
    # Only return results if there are any problems
    if (len(result["unknown_modes"]) == 0 and 
        len(result["missing_inferred_sections"]) == 0 and
        len(result["prefixed_modes"]) == 0):
        return None
    
    return result

def main():
    args = parse_args()
    
    # Set up logging
    logging_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                      level=logging_level, stream=sys.stdout)
    
    # Get time query based on specified date range
    time_query = get_time_query(args)
    
    # Get list of users to check
    users = get_users_to_check(args)
    
    logging.info(f"Checking {len(users)} users for problematic modes")
    
    # Track results by user
    problematic_users = {}
    
    # Track overall stats
    stats = {
        "total_users": len(users),
        "users_with_problems": 0,
        "total_unknown_modes": 0,
        "total_missing_sections": 0,
        "total_prefixed_modes": 0
    }
    
    for i, user_id in enumerate(users):
        if i % 10 == 0:
            logging.info(f"Checking user {i+1}/{len(users)}")
            
        result = find_problematic_modes(user_id, time_query)
        
        if result:
            # Update stats
            stats["users_with_problems"] += 1
            stats["total_unknown_modes"] += len(result["unknown_modes"])
            stats["total_missing_sections"] += len(result["missing_inferred_sections"])
            stats["total_prefixed_modes"] += len(result["prefixed_modes"])
            
            # Store detailed results
            problematic_users[str(user_id)] = result
    
    # Output results
    if args.output_json:
        output = {
            "stats": stats,
            "users": problematic_users
        }
        print(json.dumps(output, indent=2))
    else:
        print("\n===== SUMMARY OF PROBLEMATIC MODES =====")
        print(f"Total users checked: {stats['total_users']}")
        print(f"Users with problems: {stats['users_with_problems']}")
        print(f"Total unknown modes: {stats['total_unknown_modes']}")
        print(f"Total missing inferred sections: {stats['total_missing_sections']}")
        print(f"Total sections with prefixed modes: {stats['total_prefixed_modes']}")
        
        if stats["users_with_problems"] > 0:
            print("\n===== USERS WITH PROBLEMATIC MODES =====")
            problem_counts = {}
            for user_id, problems in problematic_users.items():
                unknown = len(problems["unknown_modes"])
                missing = len(problems["missing_inferred_sections"])
                prefixed = len(problems["prefixed_modes"])
                total = unknown + missing + prefixed
                problem_counts[user_id] = {
                    "unknown": unknown,
                    "missing": missing,
                    "prefixed": prefixed,
                    "total": total
                }
            
            # Create DataFrame for nicer display
            df = pd.DataFrame.from_dict(problem_counts, orient='index')
            df.index.name = 'User ID'
            df = df.sort_values('total', ascending=False)
            print(df)
            
            print("\nTo fix issues for a specific user:")
            print(f"python bin/fix_mode_inference.py --user_id <user_id>")

if __name__ == '__main__':
    main() 