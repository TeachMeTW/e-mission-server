from __future__ import unicode_literals, print_function, division, absolute_import
import unittest
import uuid
import logging
import json
import os
import time

from builtins import *
from future import standard_library
standard_library.install_aliases()

# Standard imports
import emission.storage.json_wrappers as esj

# Our imports
import emission.core.get_database as edb
import emission.storage.timeseries.timequery as estt
import emission.storage.timeseries.abstract_timeseries as esta
import emission.storage.decorations.analysis_timeseries_queries as esda
import emission.analysis.result.user_stat as eaurs
import emission.core.wrapper.user as ecwu

# Test imports
import emission.tests.common as etc


class TestUserStats(unittest.TestCase):
    def setUp(self):
        """
        Set up the test environment by loading real example data for both Android and iOS users.
        """
        # Configure logging for the test
        etc.configLogging()

        # Set analysis configuration
        self.analysis_conf_path = etc.set_analysis_config("intake.cleaning.filter_accuracy.enable", True)

        # Setup Android real example
        etc.setupRealExample(self, "emission/tests/data/real_examples/shankari_2015-aug-27")
        self.androidUUID = self.testUUID

        # Setup iOS real example
        self.testUUID = uuid.UUID("c76a0487-7e5a-3b17-a449-47be666b36f6")
        with open("emission/tests/data/real_examples/iphone_2015-11-06") as fp:
            self.entries = json.load(fp, object_hook=esj.wrapped_object_hook)
        etc.setupRealExampleWithEntries(self)
        self.iosUUID = self.testUUID

        # Apply filter accuracy on iOS UUID
        import emission.analysis.intake.cleaning.filter_accuracy as eaicf
        eaicf.filter_accuracy(self.iosUUID)

        logging.debug("androidUUID = %s, iosUUID = %s" % (self.androidUUID, self.iosUUID))

    def tearDown(self):
        """
        Clean up the test environment by removing analysis configuration and deleting test data from databases.
        """
        # Remove the analysis configuration file
        os.remove(self.analysis_conf_path)

        # Delete all time series entries for Android and iOS users
        tsdb = edb.get_timeseries_db()
        tsdb.delete_many({"user_id": self.androidUUID})
        tsdb.delete_many({"user_id": self.iosUUID})

        # Delete all pipeline state entries for Android and iOS users
        pipeline_db = edb.get_pipeline_state_db()
        pipeline_db.delete_many({"user_id": self.androidUUID})
        pipeline_db.delete_many({"user_id": self.iosUUID})

        # Delete all analysis time series entries for Android and iOS users
        analysis_ts_db = edb.get_analysis_timeseries_db()
        analysis_ts_db.delete_many({"user_id": self.androidUUID})
        analysis_ts_db.delete_many({"user_id": self.iosUUID})

        # Delete user profiles
        profile_db = edb.get_profile_db()
        profile_db.delete_one({"user_id": str(self.androidUUID)})
        profile_db.delete_one({"user_id": str(self.iosUUID)})

    def testGetAndStoreUserStatsAndroid(self):
        """
        Test get_and_store_user_stats for the Android user to ensure that user statistics
        are correctly aggregated and stored in the user profile.
        """
        # Invoke the function to get and store user stats
        eaurs.get_and_store_user_stats(str(self.androidUUID), "analysis/composite_trip")

        # Retrieve the updated user profile from the database
        profile = edb.get_profile_db().find_one({"user_id": str(self.androidUUID)})

        # Ensure that the profile exists
        self.assertIsNotNone(profile, "User profile should exist after storing stats.")

        # Verify that the expected fields are present
        self.assertIn("total_trips", profile, "User profile should contain 'total_trips'.")
        self.assertIn("labeled_trips", profile, "User profile should contain 'labeled_trips'.")
        self.assertIn("pipeline_range", profile, "User profile should contain 'pipeline_range'.")
        self.assertIn("last_call_ts", profile, "User profile should contain 'last_call_ts'.")

        expected_total_trips = -
        expected_labeled_trips = -

        self.assertEqual(profile["total_trips"], expected_total_trips,
                         f"Expected total_trips to be {expected_total_trips}, got {profile['total_trips']}")
        self.assertEqual(profile["labeled_trips"], expected_labeled_trips,
                         f"Expected labeled_trips to be {expected_labeled_trips}, got {profile['labeled_trips']}")

        # Verify pipeline range
        pipeline_range = profile.get("pipeline_range", {})
        self.assertIn("start_ts", pipeline_range, "Pipeline range should contain 'start_ts'.")
        self.assertIn("end_ts", pipeline_range, "Pipeline range should contain 'end_ts'.")

        expected_start_ts = -
        expected_end_ts = -

        self.assertEqual(pipeline_range["start_ts"], expected_start_ts,
                         f"Expected start_ts to be {expected_start_ts}, got {pipeline_range['start_ts']}")
        self.assertEqual(pipeline_range["end_ts"], expected_end_ts,
                         f"Expected end_ts to be {expected_end_ts}, got {pipeline_range['end_ts']}")

        # Verify last_call_ts
        expected_last_call_ts = -
        self.assertEqual(profile["last_call_ts"], expected_last_call_ts,
                         f"Expected last_call_ts to be {expected_last_call_ts}, got {profile['last_call_ts']}")

    def testGetAndStoreUserStatsIOS(self):
        """
        Test get_and_store_user_stats for the iOS user to ensure that user statistics
        are correctly aggregated and stored in the user profile.
        """
        # Invoke the function to get and store user stats
        eaurs.get_and_store_user_stats(str(self.iosUUID), "analysis/composite_trip")

        # Retrieve the updated user profile from the database
        profile = edb.get_profile_db().find_one({"user_id": str(self.iosUUID)})

        # Ensure that the profile exists
        self.assertIsNotNone(profile, "User profile should exist after storing stats.")

        # Verify that the expected fields are present
        self.assertIn("total_trips", profile, "User profile should contain 'total_trips'.")
        self.assertIn("labeled_trips", profile, "User profile should contain 'labeled_trips'.")
        self.assertIn("pipeline_range", profile, "User profile should contain 'pipeline_range'.")
        self.assertIn("last_call_ts", profile, "User profile should contain 'last_call_ts'.")

        expected_total_trips = -
        expected_labeled_trips = -

        self.assertEqual(profile["total_trips"], expected_total_trips,
                         f"Expected total_trips to be {expected_total_trips}, got {profile['total_trips']}")
        self.assertEqual(profile["labeled_trips"], expected_labeled_trips,
                         f"Expected labeled_trips to be {expected_labeled_trips}, got {profile['labeled_trips']}")

        # Verify pipeline range
        pipeline_range = profile.get("pipeline_range", {})
        self.assertIn("start_ts", pipeline_range, "Pipeline range should contain 'start_ts'.")
        self.assertIn("end_ts", pipeline_range, "Pipeline range should contain 'end_ts'.")

        expected_start_ts = -
        expected_end_ts = -

        self.assertEqual(pipeline_range["start_ts"], expected_start_ts,
                         f"Expected start_ts to be {expected_start_ts}, got {pipeline_range['start_ts']}")
        self.assertEqual(pipeline_range["end_ts"], expected_end_ts,
                         f"Expected end_ts to be {expected_end_ts}, got {pipeline_range['end_ts']}")

        # Verify last_call_ts
        expected_last_call_ts = -
        self.assertEqual(profile["last_call_ts"], expected_last_call_ts,
                         f"Expected last_call_ts to be {expected_last_call_ts}, got {profile['last_call_ts']}")


    def testEmptyCall(self):
        """
        Test get_and_store_user_stats with a dummy user UUID to ensure that it doesn't raise exceptions.
        """
        dummyUserId = uuid.uuid4()
        try:
            eaurs.get_and_store_user_stats(str(dummyUserId), "analysis/composite_trip")
        except Exception as e:
            self.fail(f"get_and_store_user_stats raised an exception with dummy UUID: {e}")

    def testUpdateUserProfile(self):
        """
        Test the update_user_profile function directly to ensure it correctly updates user profiles.
        """
        # Define sample data to update
        update_data = {
            "total_trips": 10,
            "labeled_trips": 7,
            "pipeline_range": {
                "start_ts": 1609459200,  # 2021-01-01 00:00:00
                "end_ts": 1609545600     # 2021-01-02 00:00:00
            },
            "last_call_ts": 1609632000  # 2021-01-03 00:00:00
        }

        # Invoke the function to update the user profile
        eaurs.update_user_profile(str(self.androidUUID), update_data)

        # Retrieve the updated user profile from the database
        profile = edb.get_profile_db().find_one({"user_id": str(self.androidUUID)})

        # Ensure that the profile exists
        self.assertIsNotNone(profile, "User profile should exist after updating.")

        # Verify that the profile contains the updated data
        for key, value in update_data.items():
            self.assertIn(key, profile, f"User profile should contain '{key}'.")
            self.assertEqual(profile[key], value,
                             f"Expected '{key}' to be {value}, got {profile[key]}.")


if __name__ == '__main__':
    unittest.main()
