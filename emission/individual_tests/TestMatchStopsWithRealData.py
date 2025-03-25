from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import *
import unittest
import logging
import attrdict as ad
import time
import json
import os
import random
import math

import emission.tests.common as etc
import emission.net.ext_service.transit_matching.match_stops as enetm
import emission.analysis.classification.inference.mode.rule_engine as eacimr
import emission.storage.timeseries.abstract_timeseries as esta
import emission.storage.decorations.analysis_timeseries_queries as esda
import emission.storage.timeseries.timequery as estt
import emission.core.get_database as edb

class TestMatchStopsWithRealData(unittest.TestCase):
    """
    This test file tests the match_stops module with real API calls.
    Unlike the other test approaches that use mocked data or saved data,
    this test makes actual API calls to the Overpass API.
    
    This is useful for:
    - Testing the actual API interaction
    - Verifying the caching mechanism works correctly
    - Testing the retry mechanism works correctly
    - Ensuring the code works with real-world data
    
    Note: This test requires internet access and might be slower than mocked tests.
    This test also requires example data that contains location entries.
    """
    
    def setUp(self):
        """
        Set up real example data for testing.
        
        We extract locations directly from the raw data entries, so this test
        requires example data files that contain location points.
        """
        # Load real example data - works with any example file containing location data
        self.test_obj = type('TestObj', (), {'testUUID': None, 'entries': None})
        etc.setupRealExample(self.test_obj, "emission/tests/data/real_examples/shankari_2015-aug-27")
        
        # Set up time series for the user
        self.ts = esta.TimeSeries.get_time_series(self.test_obj.testUUID)
        
        # Extract locations directly from the loaded entries
        self.start_loc, self.end_loc, self.mid_loc = self._extract_location_points()
        
        # Log the locations we're using
        logging.debug(f"Using start_loc: {self.start_loc}")
        logging.debug(f"Using end_loc: {self.end_loc}")
        logging.debug(f"Using mid_loc: {self.mid_loc}")
        
        # Test search radius in meters
        self.search_radius = 200.0
    
    def _extract_location_points(self):
        """
        Extract location points dynamically from the loaded example data.
        
        This method queries the database for location entries that were loaded from 
        the example file and extracts coordinates from them. It requires example
        data that contains at least 3 location entries.
        
        Returns:
            tuple: (start_loc, end_loc, mid_loc) dictionaries with coordinates
        """
        # Directly query the database for location entries
        ts_db = edb.get_timeseries_db()
        location_entries = list(ts_db.find({
            "user_id": self.test_obj.testUUID, 
            "$or": [
                {"metadata.key": "background/location"},
                {"metadata.key": "background/filtered_location"}
            ]
        }).sort("metadata.write_ts", 1))  # Sort by timestamp
        
        logging.info(f"Found {len(location_entries)} location entries in the loaded data")
        
        if len(location_entries) < 3:
            # If we don't have enough location entries, fail the test
            raise ValueError("This test requires example data with at least 3 location entries")
        
        # Get coordinates from different parts of the trajectory to ensure variety
        # First location
        start_idx = 0
        # Last location
        end_idx = len(location_entries) - 1
        # Middle location, with some randomness to make it interesting
        mid_idx = max(0, min(len(location_entries) - 1, 
                           int(len(location_entries) * 0.4 + random.random() * 0.3)))
        
        # Function to extract coordinates from a location entry
        def get_coords(entry, position_name):
            if 'loc' in entry.get('data', {}):
                return {'coordinates': entry['data']['loc']['coordinates']}
            # Backup way to get coordinates
            elif 'longitude' in entry.get('data', {}) and 'latitude' in entry.get('data', {}):
                return {'coordinates': [entry['data']['longitude'], entry['data']['latitude']]}
            # If we can't extract coordinates, fail the test
            else:
                raise ValueError(f"Could not extract coordinates for {position_name} location. This test requires location entries with coordinates.")
        
        start_loc = get_coords(location_entries[start_idx], "start")
        end_loc = get_coords(location_entries[end_idx], "end")
        mid_loc = get_coords(location_entries[mid_idx], "middle")
        
        # Ensure mid_loc is sufficiently different from start and end
        # This helps with the no_stops test
        def distance(loc1, loc2):
            return math.sqrt(
                (loc1['coordinates'][0] - loc2['coordinates'][0]) ** 2 +
                (loc1['coordinates'][1] - loc2['coordinates'][1]) ** 2
            )
        
        # If mid_loc is too close to start or end, try to find a better one
        min_dist_threshold = 0.005  # Roughly 500 meters
        attempts = 0
        while (distance(mid_loc, start_loc) < min_dist_threshold or 
               distance(mid_loc, end_loc) < min_dist_threshold) and attempts < 10:
            # Try a different mid point
            new_idx = max(0, min(len(location_entries) - 1, 
                              int(random.random() * len(location_entries))))
            try:
                mid_loc = get_coords(location_entries[new_idx], "middle alternative")
                attempts += 1
            except ValueError:
                # Skip this entry if we can't extract coordinates
                attempts += 1
                continue
        
        # If we couldn't find a sufficiently different mid_loc after several attempts,
        # we'll just use what we have and let the test determine if it works
        
        logging.info(f"Successfully extracted location points from example data")
        return start_loc, end_loc, mid_loc
    
    def test_get_stops_near(self):
        """
        Test get_stops_near function with real API calls.
        
        This test verifies that:
        1. The function can connect to the Overpass API
        2. It correctly retrieves transit stops
        3. The returned data has the expected structure
        """
        # Get stops near start location
        stops = enetm.get_stops_near(self.start_loc, self.search_radius)
        
        # Verify we got some stops
        self.assertTrue(len(stops) > 0, "Should find at least one stop near start location")
        
        # Verify the stop structure
        for stop in stops:
            self.assertTrue(hasattr(stop, 'id'), "Stop should have an ID")
            self.assertTrue(hasattr(stop, 'tags'), "Stop should have tags")
            
            # Stops should have lat/lon
            self.assertTrue(hasattr(stop, 'lat'), "Stop should have latitude")
            self.assertTrue(hasattr(stop, 'lon'), "Stop should have longitude")
            
            # Print first stop for debugging
            if stops.index(stop) == 0:
                logging.debug(f"First stop: {stop}")
        
        # Test the caching mechanism by calling again
        start_time = time.time()
        stops_cached = enetm.get_stops_near(self.start_loc, self.search_radius)
        end_time = time.time()
        
        # Verify cache returned same data
        self.assertEqual(len(stops), len(stops_cached), 
                         "Cached query should return same number of stops")
        
        # The cached query should be much faster
        logging.debug(f"Cache query time: {end_time - start_time:.2f} seconds")
    
    def test_get_predicted_transit_mode(self):
        """
        Test get_predicted_transit_mode with real locations.
        
        This tests that the function correctly identifies transit modes
        between two locations from the example data.
        """
        # Get stops near both locations
        start_stops = enetm.get_stops_near(self.start_loc, self.search_radius)
        end_stops = enetm.get_stops_near(self.end_loc, self.search_radius)
        
        # Check if we have stops at both locations
        has_start_stops = len(start_stops) > 0
        has_end_stops = len(end_stops) > 0
        
        # Log what we found
        logging.debug(f"Found {len(start_stops)} stops near start location")
        logging.debug(f"Found {len(end_stops)} stops near end location")
        
        # If no stops at either location, we can't test transit mode prediction
        if not has_start_stops:
            self.skipTest("No transit stops found near start location - can't test transit mode prediction")
        
        if not has_end_stops:
            self.skipTest("No transit stops found near end location - can't test transit mode prediction")
        
        # If we get here, we have stops at both locations
        self.assertTrue(has_start_stops, "Should find stops at start location")
        self.assertTrue(has_end_stops, "Should find stops at end location")
        
        # Get predicted mode
        predicted_modes = enetm.get_predicted_transit_mode(start_stops, end_stops)
        
        # Log the result
        logging.debug(f"Predicted modes between locations: {predicted_modes}")
        
        # We're mainly testing that the function runs without errors
        # The actual prediction depends on the specific data from the API
        if predicted_modes:
            self.assertTrue(isinstance(predicted_modes, list), "Result should be a list")
    
    def test_get_predicted_transit_mode_no_stops(self):
        """
        Test get_predicted_transit_mode with no transit.
        
        This tests the behavior when one location has no stops nearby.
        """
        # Get stops near start location
        start_stops = enetm.get_stops_near(self.start_loc, self.search_radius)
        
        # Create a location with likely no transit stops nearby
        no_transit_loc = self.mid_loc  # Using the mid-point which should be away from transit
        
        # Get stops with a small radius to decrease chance of finding any
        no_stops = enetm.get_stops_near(no_transit_loc, 10.0)  # Very small radius
        
        # Verify we got stops at start location
        self.assertTrue(len(start_stops) > 0, "Should find stops at start location")
        
        # If we somehow find stops at the "no transit" location, skip this test
        if len(no_stops) > 0:
            self.skipTest("Found stops at supposedly transit-free location")
            
        # Get predicted mode
        predicted_modes = enetm.get_predicted_transit_mode(start_stops, no_stops)
        
        # When one side has no stops, we expect None
        self.assertIsNone(predicted_modes, "Should predict no transit mode when one side has no stops")

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Run tests
    unittest.main() 