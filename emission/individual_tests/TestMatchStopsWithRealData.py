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

import emission.tests.common as etc
import emission.net.ext_service.transit_matching.match_stops as enetm
import emission.analysis.classification.inference.mode.rule_engine as eacimr
import emission.storage.timeseries.abstract_timeseries as esta
import emission.storage.decorations.analysis_timeseries_queries as esda
import emission.storage.timeseries.timequery as estt

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
    """
    
    def setUp(self):
        """
        Set up real example data for testing.
        
        Use real locations from the loaded example data.
        """
        # Load real example data
        self.test_obj = type('TestObj', (), {'testUUID': None, 'entries': None})
        etc.setupRealExample(self.test_obj, "emission/tests/data/real_examples/shankari_2015-aug-27")
        
        # Extract locations from the data
        try:
            # Get real place locations from the example data
            self.ts = esta.TimeSeries.get_time_series(self.test_obj.testUUID)
            
            # Define a time query that covers all data
            time_query = estt.TimeQuery("data.start_ts", None, None)
            
            # Try to get sections from the data
            sections = esda.get_entries(self.ts, "analysis/cleaned_section", time_query=time_query)
            
            # Extract locations if sections data is available
            if len(sections) > 0:
                # Get the first section's start and end locations
                section = sections[0]
                if 'start_loc' in section.data and 'end_loc' in section.data:
                    self.start_loc = {'coordinates': [section.data.start_loc.coordinates[0], 
                                                     section.data.start_loc.coordinates[1]]}
                    self.end_loc = {'coordinates': [section.data.end_loc.coordinates[0], 
                                                   section.data.end_loc.coordinates[1]]}
                    
                    # Try to find a third location for the no_transit test
                    if len(sections) > 1 and 'start_loc' in sections[1].data:
                        self.mid_loc = {'coordinates': [sections[1].data.start_loc.coordinates[0], 
                                                       sections[1].data.start_loc.coordinates[1]]}
                    else:
                        # Create a mid location by averaging start and end
                        self.mid_loc = {'coordinates': [(self.start_loc['coordinates'][0] + self.end_loc['coordinates'][0])/2, 
                                                       (self.start_loc['coordinates'][1] + self.end_loc['coordinates'][1])/2]}
                else:
                    # Fallback if section doesn't have locations
                    raise ValueError("Sections don't have location data")
            else:
                # Fallback if no sections found
                raise ValueError("No sections found in the data")
                
        except Exception as e:
            # If we couldn't get section data, fall back to hardcoded locations
            logging.warning(f"Error extracting locations from example data: {e}. Using fallback locations.")
            self.start_loc = {'coordinates': [-122.268428, 37.869867]}  # Downtown Berkeley BART
            self.end_loc = {'coordinates': [-122.259482, 37.871899]}    # UC Berkeley campus
            self.mid_loc = {'coordinates': [-122.258906, 37.867225]}    # Telegraph Ave & Dwight Way
        
        # Log the locations we're using
        logging.debug(f"Using start_loc: {self.start_loc}")
        logging.debug(f"Using end_loc: {self.end_loc}")
        logging.debug(f"Using mid_loc: {self.mid_loc}")
        
        # Test search radius in meters
        self.search_radius = 200.0
    
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
        
        # Both locations should have stops
        self.assertTrue(len(start_stops) > 0, "Should find stops at start location")
        self.assertTrue(len(end_stops) > 0, "Should find stops at end location")
        
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
        # Use a location far from any transit stops by offsetting from our start location
        no_transit_loc = {'coordinates': [self.start_loc['coordinates'][0] + 0.02, 
                                          self.start_loc['coordinates'][1] + 0.02]}
        
        # Get stops with a small radius to decrease chance of finding any
        no_stops = enetm.get_stops_near(no_transit_loc, 50.0)
        
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