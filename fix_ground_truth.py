import json
import os
import attrdict as ad
import emission.core.wrapper.localdate as ecwl
import emission.storage.json_wrappers as esj
import emission.analysis.plotting.geojson.geojson_feature_converter as gfc
import emission.tests.common as etc
import emission.pipeline.reset as epr
import emission.tests.pipelineTests.TestPipelineReset as tpr
import unittest
import time

# Set up test environment
dataFile_1 = 'emission/tests/data/real_examples/shankari_2016-07-22'
dataFile_2 = 'emission/tests/data/real_examples/shankari_2016-07-25'
start_ld_1 = ecwl.LocalDate({'year': 2016, 'month': 7, 'day': 22})
start_ld_2 = ecwl.LocalDate({'year': 2016, 'month': 7, 'day': 25})
etc.set_analysis_config('analysis.result.section.key', 'analysis/cleaned_section')

# Create a test class from TestPipelineReset
test = tpr.TestPipelineReset()
test.setUp()  # This will set the random seed and analysis config

# Run the test steps manually
# Load all data
etc.setupRealExample(test, dataFile_1)
etc.runIntakePipeline(test.testUUID)
test.entries = json.load(open(dataFile_2), object_hook=esj.wrapped_object_hook)
etc.setupRealExampleWithEntries(test)
etc.runIntakePipeline(test.testUUID)

# Get the API results
api_result_1 = gfc.get_geojson_for_dt(test.testUUID, start_ld_1, start_ld_1)
api_result_2 = gfc.get_geojson_for_dt(test.testUUID, start_ld_2, start_ld_2)

# Save them as ground truths
with open(dataFile_1 + ".ground_truth", "w") as outfile:
    wrapped_gt = {
        "data": api_result_1,
        "metadata": {
            "key": f"diary/trips-2016-07-22",
            "type": "document",
            "write_ts": int(time.time())
        }
    }
    json.dump(wrapped_gt, outfile, indent=4, default=esj.wrapped_default)

with open(dataFile_2 + ".ground_truth", "w") as outfile:
    wrapped_gt = {
        "data": api_result_2,
        "metadata": {
            "key": f"diary/trips-2016-07-25",
            "type": "document",
            "write_ts": int(time.time())
        }
    }
    json.dump(wrapped_gt, outfile, indent=4, default=esj.wrapped_default)

print(f"Ground truth files updated successfully")
print(f"Day 1 has {len(api_result_1)} trips")
print(f"Day 2 has {len(api_result_2)} trips")

# Clean up
test.tearDown()
etc.clear_analysis_config() 