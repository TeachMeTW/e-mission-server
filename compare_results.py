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
ground_truth_1 = json.load(open(dataFile_1+".ground_truth"), object_hook=esj.wrapped_object_hook)
ground_truth_2 = json.load(open(dataFile_2+".ground_truth"), object_hook=esj.wrapped_object_hook)

# Run both pipelines
etc.setupRealExample(test, dataFile_1)
etc.runIntakePipeline(test.testUUID)
test.entries = json.load(open(dataFile_2), object_hook=esj.wrapped_object_hook)
etc.setupRealExampleWithEntries(test)
etc.runIntakePipeline(test.testUUID)

# Check results: so far, so good
api_result = gfc.get_geojson_for_dt(test.testUUID, start_ld_1, start_ld_1)
print(f'API Result for day 1 has {len(api_result)} trips:')
for i, trip in enumerate(api_result):
    print(f'Trip {i+1}: {trip["properties"]["start_fmt_time"]} -> {trip["properties"]["end_fmt_time"]} sections:{len(trip["features"])}')

print(f'Ground Truth for day 1 has {len(ground_truth_1["data"])} trips:')
for i, trip in enumerate(ground_truth_1['data']):
    print(f'Trip {i+1}: {trip["properties"]["start_fmt_time"]} -> {trip["properties"]["end_fmt_time"]} sections:{len(trip["features"])}')

# Clean up
test.tearDown()
etc.clear_analysis_config() 