from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
# Standard imports
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import *
import logging
from datetime import datetime, timedelta
import json
import os
import emission.storage.json_wrappers as esj
import uuid
import pymongo

# Our imports
import emission.core.get_database as edb
from emission.core.get_database import get_client_db, get_section_db
import emission.core.get_database as edb
import emission.core.wrapper.user as ecwu

def makeValid(client):
  client.clientJSON['start_date'] = str(datetime.now() + timedelta(days=-2))
  client.clientJSON['end_date'] = str(datetime.now() + timedelta(days=+2))
  # print (client.clientJSON)
  client._Client__update(client.clientJSON)

def makeExpired(client):
  client.clientJSON['start_date'] = str(datetime.now() + timedelta(days=-4))
  client.clientJSON['end_date'] = str(datetime.now() + timedelta(days=-2))
  # print (client.clientJSON)
  client._Client__update(client.clientJSON)

def updateUserCreateTime(uuid):
  from emission.core.wrapper.user import User
  
  user = User.fromUUID(uuid)
  user.changeUpdateTs(timedelta(days = -20))

def dropAllCollections(db):
  collections = db.list_collection_names()
  print("collections = %s" % collections)
  for coll in collections:
    if coll.startswith('system'):
      print("Skipping system collection %s" % coll)
    else: 
      print("Dropping collection %s" % coll)
      db.drop_collection(coll)
 
  import emission.storage.timeseries.builtin_timeseries as bits
  bits.ts_enum_map = bits._get_enum_map()
  # we expect to see [7, 51]
  print(f"After restoring indices on cached collections, we see {[len(bits.ts_enum_map[k].index_information().keys()) for k in bits.ts_enum_map]}")


def purgeSectionData(Sections, userName):
    """
    Deletes all sections for this user.
    TODO: Need to extend it to delete entries across all collections
    """
    Sections.delete_many({'user_id' : userName})

def loadTable(serverName, tableName, fileName):
  tableColl = edb._get_current_db()[tableName]
  with open(fileName) as fp:
      dataJSON = json.load(fp)
  for row in dataJSON:
    tableColl.insert_one(row)

# Create a dummy section with the main stuff that we use in our code
def createDummySection(startTime, endTime, startLoc, endLoc, predictedMode = None, confirmedMode = None):
  section = {
             'source': 'Shankari',
             'section_start_datetime': startTime,
             'section_end_datetime': endTime,
             'section_start_time': startTime.isoformat(),
             'section_end_time': endTime.isoformat(),
             'section_start_point': {'type': 'Point', 'coordinates': startLoc},
             'section_end_point': {'type': 'Point', 'coordinates': endLoc},
            }
  if predictedMode != None:
    section['predicted_mode'] = predictedMode
  if confirmedMode != None:
    section['confirmed_mode'] = confirmedMode

  get_section_db().insert_one(section)
  return section

def updateSections(testCase):
    from emission.core.wrapper.user import User
    """
    Updates sections with appropriate test data
    Should be called anytime new data is loaded into the
    'Stage_Sections' table
    """
    testCase.uuid_list = []
    for section in testCase.SectionsColl.find():
      section['section_start_datetime'] = testCase.dayago
      section['section_end_datetime'] = testCase.dayago + timedelta(hours = 1)
      section['predicted_mode'] = [0, 0.4, 0.6, 0]
      section['confirmed_mode'] = ''
      # Replace the user email with the UUID
      curr_uuid = User.fromEmail(section['user_id']).uuid
      section['user_id'] = curr_uuid
      testCase.uuid_list.append(curr_uuid)
      testCase.SectionsColl.save(section)

def getRealExampleEmail(testObj):
    return testObj.branch + "_" + testObj._testMethodName

def fillExistingUUID(testObj):
    userObj = ecwu.User.fromEmail(getRealExampleEmail(testObj))
    print("Setting testUUID to %s" % userObj.uuid)
    testObj.testUUID = userObj.uuid

def getRegEmailIfPresent(testObj):
    if hasattr(testObj, "evaluation") and testObj.evaluation:
        logging.info("evaluation, returning email = %s" % reg_email)
        reg_email = getRealExampleEmail(testObj)
        return reg_email
    elif hasattr(testObj, "testEmail"):
        return testObj.testEmail
    else:
        return None

def createAndFillUUID(testObj):
    regEmail = getRegEmailIfPresent(testObj)
    if regEmail is not None:
        logging.info("registering email = %s" % regEmail)
        user = ecwu.User.register(regEmail)
        testObj.testUUID = user.uuid
    else:
        logging.info("No reg email found, not registering email")
        testObj.testUUID = uuid.uuid4()
    # ensure that there is unprocessed data so that the pipeline will actually be run
    edb.get_profile_db().update_one({"user_id": testObj.testUUID}, {"$set": {"last_location_ts": 3650, "pipeline_range": {"end_ts": None}}}, upsert=True)

def setupRealExample(testObj, dump_file):
    logging.info("Before loading from %s, timeseries db size = %s" %
        (dump_file, edb.get_timeseries_db().estimated_document_count()))
    with open(dump_file) as dfp:
        testObj.entries = json.load(dfp, object_hook = esj.wrapped_object_hook)
        createAndFillUUID(testObj)
        print("Setting up real example for %s" % testObj.testUUID)
        setupRealExampleWithEntries(testObj)

def setupRealExampleWithEntries(testObj):
    tsdb = edb.get_timeseries_db()
    for entry in testObj.entries:
        entry["user_id"] = testObj.testUUID
        # print "Saving entry with write_ts = %s and ts = %s" % (entry["metadata"]["write_fmt_time"],
        #                                                        entry["data"]["fmt_time"])
        edb.save(tsdb, entry)
        
    logging.info("After loading, timeseries db size = %s" % edb.get_timeseries_db().estimated_document_count())
    logging.debug("First few entries = %s" % 
                    [e["data"]["fmt_time"] if "fmt_time" in e["data"] else e["metadata"]["write_fmt_time"] for e in 
                        list(edb.get_timeseries_db().find({"user_id": testObj.testUUID}).sort("data.write_ts",
                                                                                       pymongo.ASCENDING).limit(10))])

def setupIncomingEntries():
    with open("emission/tests/data/netTests/android.activity.txt") as aaef:
        activity_entry = json.load(aaef)
    with open("emission/tests/data/netTests/android.location.txt") as alef:
        location_entry = json.load(alef)
    with open("emission/tests/data/netTests/android.transition.txt") as atef:
        transition_entry = json.load(atef)
    entry_list = [activity_entry, location_entry, transition_entry]

    with open("emission/tests/data/netTests/ios.activity.txt") as iaef:
        ios_activity_entry = json.load(iaef)
    with open("emission/tests/data/netTests/ios.location.txt") as ilef:
        ios_location_entry = json.load(ilef)
    with open("emission/tests/data/netTests/ios.transition.txt") as itef:
        ios_transition_entry = json.load(itef)

    ios_entry_list = [ios_activity_entry, ios_location_entry, ios_transition_entry]

    return (entry_list, ios_entry_list)

def restoreOriginalEnvVars(originalEnvVars, modifiedEnvVars):
    for env_var_name, env_var_value in modifiedEnvVars.items():
        del os.environ[env_var_name]
    # Restoring original db environment variables
    for env_var_name, env_var_value in originalEnvVars.items():
        os.environ[env_var_name] = env_var_value

def runIntakePipeline(uuid):
    # Move these imports here so that we don't inadvertently load the modules,
    # and any related config modules, before we want to
    import emission.analysis.userinput.matcher as eaum
    import emission.analysis.intake.cleaning.filter_accuracy as eaicf
    import emission.storage.timeseries.format_hacks.move_filter_field as estfm
    import emission.analysis.intake.segmentation.trip_segmentation as eaist
    import emission.analysis.intake.segmentation.section_segmentation as eaiss
    import emission.analysis.intake.cleaning.location_smoothing as eaicl
    import emission.analysis.intake.cleaning.clean_and_resample as eaicr
    import emission.analysis.classification.inference.mode.pipeline as eacimp
    import emission.analysis.userinput.expectations as eaue
    import emission.analysis.classification.inference.labels.pipeline as eacilp
    import emission.analysis.plotting.composite_trip_creation as eapcc
    import emission.analysis.result.user_stat as eaurs

    eaum.match_incoming_user_inputs(uuid)
    eaicf.filter_accuracy(uuid)
    eaist.segment_current_trips(uuid)
    eaiss.segment_current_sections(uuid)
    eaicl.filter_current_sections(uuid)
    eaicr.clean_and_resample(uuid)
    eacimp.predict_mode(uuid)
    eacilp.infer_labels(uuid)
    eaue.populate_expectations(uuid)
    eaum.create_confirmed_objects(uuid)
    eapcc.create_composite_objects(uuid)
    eaurs.get_and_store_pipeline_dependent_user_stats(uuid, "analysis/composite_trip")

def configLogging():
    """
    Standard function to be called from the test cases to turn on logging.
    We really want the tests to configure logging in their main method so
    that individual tests can be run when they fail. But we also want a standard
    method that we can change quickly and easily.

    This is a simple way to meet both requirements.

    :return: None
    """
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(thread)d:%(message)s',
                    level=logging.DEBUG)

def setupTokenListAuth(self):
    token_list_conf_file = open(self.token_list_conf_path, "w")
    token_list_conf_json = {
        "token_list": self.token_list_path
    }

    token_list_conf_file.write(str(json.dumps(token_list_conf_json)))
    token_list_conf_file.close()
    token_list_file = open(self.token_list_path, "w")
    token_list_file.write("correct_horse_battery_staple\n")
    token_list_file.write("collar_highly_asset_ovoid_sultan\n")
    token_list_file.write("caper_hangup_addle_oboist_scroll\n")
    token_list_file.write("couple_honcho_abbot_obtain_simple\n")
    token_list_file.close()

def tearDownTokenListAuth(self):
    import os

    os.remove(self.token_list_conf_path)
    os.remove(self.token_list_path)

def createDummyRequestEnviron(self, addl_headers, request_body):
    # request_body is a StringIO object
    test_environ = {'HTTP_REFERER': 'http://localhost:8080/',
        'SERVER_SOFTWARE': 'CherryPy/3.6.0 Server',
        'SCRIPT_NAME': '',
        'ACTUAL_SERVER_PROTOCOL': 'HTTP/1.1',
        'REQUEST_METHOD': 'POST',
        'PATH_INFO': '/result/heatmap/pop.route/local_date',
        'SERVER_PROTOCOL': 'HTTP/1.1',
        'QUERY_STRING': '',
        'bottle.request.body': request_body,
        'CONTENT_TYPE': 'application/json;charset=utf-8',
        'wsgi.input': request_body,
        'wsgi.multithread': True,
        'HTTP_ACCEPT_LANGUAGE': 'en-US,en;q=0.5',
        'HTTP_ACCEPT_ENCODING': 'gzip, deflate'
    }
    if addl_headers is not None:
        test_environ.update(addl_headers)
    return test_environ

def set_analysis_config(key, value):
    """
    Tests that call this in their setUp must call clear_analysis_config in their tearDown
    """
    import emission.analysis.config as eac
    import shutil

    shutil.copyfile(eac.ANALYSIS_CONF_DEV_PATH, eac.ANALYSIS_CONF_PATH)
    with open(eac.ANALYSIS_CONF_PATH) as fd:
        curr_config = json.load(fd)
    curr_config[key] = value
    with open(eac.ANALYSIS_CONF_PATH, "w") as fd:
        json.dump(curr_config, fd, indent=4)
    logging.debug("Finished setting up %s" % eac.ANALYSIS_CONF_PATH)
    with open(eac.ANALYSIS_CONF_PATH) as fd:
        logging.debug("Current values are %s" % json.load(fd))

    eac.reload_config()

def clear_analysis_config():
    import emission.analysis.config as eac
    if os.path.exists(eac.ANALYSIS_CONF_PATH):
        os.remove(eac.ANALYSIS_CONF_PATH)
    eac.reload_config()

def copy_dummy_seed_for_inference():
    import shutil
    import os

    seed_json_source = "emission/tests/data/seed_model_from_test_data.json"
    seed_json_dest = "seed_model.json"
    result = shutil.copyfile(seed_json_source, seed_json_dest)
    logging.debug("Copied file %s -> %s with result %s" % (seed_json_source, seed_json_dest, result))

    assert os.path.exists(seed_json_dest), "File %s not found after copy" % seed_json_dest
    return seed_json_dest
