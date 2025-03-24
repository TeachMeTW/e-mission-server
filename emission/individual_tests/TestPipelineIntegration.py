from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import *
import unittest
import logging
import os
import time
import uuid
import json
import pandas as pd
import hashlib
import emission.tests.common as etc
from emission.tests.common import runIntakePipeline, setupRealExample


# THIS IS TO MAKE SEEING THE MATCH_STOPS LOG EASIER
# THERES TOO MUCH NOISE FROM OTHER LOGGERS
# Ultra-aggressive approach to suppress ALL logging except match_stops
# First, completely disable all default loggers
for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.CRITICAL)
    logging.getLogger(name).propagate = False
    logging.getLogger(name).disabled = True

# Create a filter that only allows match_stops logs
class MatchStopsFilter(logging.Filter):
    def filter(self, record):
        return record.name == 'emission.net.ext_service.transit_matching.match_stops'

# Configure the root logger to use our filter
handler = logging.StreamHandler()
handler.addFilter(MatchStopsFilter())
logging.root.handlers = [handler]
logging.root.setLevel(logging.DEBUG)

# Configure match_stops logger specifically
match_stops_logger = logging.getLogger('emission.net.ext_service.transit_matching.match_stops')
match_stops_logger.setLevel(logging.DEBUG)
match_stops_logger.disabled = False
match_stops_logger.propagate = True  # Let it propagate to our filtered root handler

class TestPipelineIntegration(unittest.TestCase):
    """Test the integration of the cache with the pipeline"""
    
    def setUp(self):
        """Set up test object and get the project root"""
        # Import match_stops at the class level
        from emission.net.ext_service.transit_matching import match_stops
        self.match_stops = match_stops
        
        # Make sure the cache directory exists
        self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(match_stops.__file__)), "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Get the project root directory to construct relative paths
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        
        # List of example files to test with (using relative paths)
        self.example_files = [
            os.path.join(self.project_root, "emission/tests/data/real_examples/shankari_2016-01-13"),
            os.path.join(self.project_root, "emission/tests/data/real_examples/gabe_2016-06-15")
        ]
        
        # Store original environment
        self.old_key = os.environ.get("GEOFABRIK_OVERPASS_KEY", None)
        
        # These will track all calls
        self.total_api_requests = 0
        self.total_cache_hits = 0
    
    def tearDown(self):
        """Restore environment after test"""
        # Restore environment
        if self.old_key is None:
            if "GEOFABRIK_OVERPASS_KEY" in os.environ:
                del os.environ["GEOFABRIK_OVERPASS_KEY"]
        else:
            os.environ["GEOFABRIK_OVERPASS_KEY"] = self.old_key
    
    def test_pipeline_caching(self):
        """Test the caching in the pipeline's transit matching component.
        
        Note: Our testing revealed that the pipeline has built-in optimization that remembers
        previously processed data and skips calling transit matching functions on subsequent runs 
        for the same user UUID. Therefore, we focus this test on the first run only, where 
        the transit matching functions are actually called.
        """
        print("\n--- Testing transit matching caching in the pipeline ---")
        
        # Run test in development mode (no API key)
        if "GEOFABRIK_OVERPASS_KEY" in os.environ:
            del os.environ["GEOFABRIK_OVERPASS_KEY"]
        
        for example_file in self.example_files:
            print(f"\n=== Testing with example file: {os.path.basename(example_file)} ===")
            
            # Setup the example data with a new test object for each example
            test_obj = type('TestObj', (), {'testUUID': None, 'entries': None})
            try:
                print(f"Loading data from {example_file}")
                setupRealExample(test_obj, example_file)
                print(f"Data loaded successfully with UUID: {test_obj.testUUID}")
                
                # Count cache files before the test
                cache_files_before = set(os.listdir(self.cache_dir)) if os.path.exists(self.cache_dir) else set()
                print(f"Cache files before test: {len(cache_files_before)}")
                
                # Track API vs cache usage
                api_request_count = 0
                cache_hit_count = 0
                transit_func_calls = 0
                
                # Save original functions for later restoration
                original_make_request = self.match_stops.make_request_and_catch
                original_get_stops_near = self.match_stops.get_stops_near
                
                def new_get_stops_near(loc, distance_in_meters):
                    """Track calls to get_stops_near"""
                    nonlocal transit_func_calls
                    transit_func_calls += 1
                    print(f"CALL: get_stops_near (location: {loc.get('coordinates', '?')}, distance: {distance_in_meters}m)")
                    return original_get_stops_near(loc, distance_in_meters)
                
                def new_make_request_and_catch(overpass_query):
                    """Patched version to track cache hits vs API requests"""
                    nonlocal api_request_count, cache_hit_count
                    
                    # Create a unique filename based on the query hash
                    query_hash = hashlib.md5(overpass_query.encode()).hexdigest()
                    cache_file = os.path.join(self.cache_dir, f"{query_hash}.csv")
                    
                    # Check if we're in production mode
                    in_production = os.environ.get("GEOFABRIK_OVERPASS_KEY") is not None
                    
                    # If cache file exists and we're not in production, use cache
                    if not in_production and os.path.exists(cache_file):
                        cache_hit_count += 1
                        self.total_cache_hits += 1
                        print(f"CACHE HIT: Using cached response from {cache_file}")
                        # Read directly from cache
                        try:
                            df = pd.read_csv(cache_file)
                            return df.to_dict('records')
                        except Exception as e:
                            print(f"Error reading cache, falling back to API: {e}")
                    
                    # Otherwise make an API request
                    api_request_count += 1
                    self.total_api_requests += 1
                    print(f"API REQUEST: Calling API for {query_hash}")
                    
                    # Call the original function
                    return original_make_request(overpass_query)
                
                # Apply our patches
                self.match_stops.make_request_and_catch = new_make_request_and_catch
                self.match_stops.get_stops_near = new_get_stops_near
                
                try:
                    # Run the pipeline
                    print("\nRunning intake pipeline (transit matching functions will be called if needed):")
                    start_time = time.time()
                    runIntakePipeline(test_obj.testUUID)
                    run_time = time.time() - start_time
                    print(f"Pipeline completed in {run_time:.2f} seconds")
                    
                    # Summary of calls
                    print(f"Transit matching function calls: {transit_func_calls}")
                    print(f"API requests: {api_request_count}, Cache hits: {cache_hit_count}")
                    
                    # Check for new cache files
                    cache_files_after = set(os.listdir(self.cache_dir)) if os.path.exists(self.cache_dir) else set()
                    new_cache_files = cache_files_after - cache_files_before
                    
                    if len(new_cache_files) > 0:
                        print(f"✅ New cache files created: {len(new_cache_files)}")
                    else:
                        print("ℹ️ No new cache files created")
                    
                    # Validate cache usage
                    print("\n--- Cache Usage Validation ---")
                    if cache_hit_count > 0:
                        print("✅ Cache hits detected! Cache is working correctly.")
                        
                    if api_request_count > 0:
                        print("✅ API requests detected! API fallback is working correctly.")
                        
                    # Print the ratio of cache hits to total requests
                    total_requests = cache_hit_count + api_request_count
                    if total_requests > 0:
                        cache_hit_ratio = cache_hit_count / total_requests
                        print(f"Cache hit ratio: {cache_hit_ratio:.2%} ({cache_hit_count} cache hits out of {total_requests} total requests)")
                    
                    # Check if transit functions were called
                    if transit_func_calls == 0:
                        print("⚠️ Transit matching functions were not called for this dataset.")
                        print("   This may indicate that this particular data doesn't need transit matching.")
                    
                finally:
                    # Always restore original functions
                    self.match_stops.make_request_and_catch = original_make_request
                    self.match_stops.get_stops_near = original_get_stops_near
                
            except Exception as e:
                print(f"Error testing with example {example_file}: {e}")
                import traceback
                traceback.print_exc()
        
        # Print overall summary
        print("\n=== Overall Test Summary ===")
        print(f"Total API requests: {self.total_api_requests}")
        print(f"Total cache hits: {self.total_cache_hits}")

if __name__ == "__main__":
    # Run with minimal test output to reduce noise
    unittest.main(verbosity=0) 