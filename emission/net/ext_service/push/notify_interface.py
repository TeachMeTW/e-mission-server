from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
# Standard imports
from future import standard_library
standard_library.install_aliases()
from builtins import *
from builtins import object
import json
import logging
import importlib
import os

import emission.core.backwards_compat_config as ecbc

# Note that the URL is hardcoded because the API endpoints are not standardized.
# If we change a push provider, we will need to modify to match their endpoints.
# Hardcoding will remind us of this :)
# We can revisit this if push providers eventually decide to standardize...

push_config = ecbc.get_config('conf/net/ext_service/push.json',
    {"PUSH_PROVIDER": "provider", "PUSH_SERVER_AUTH_TOKEN": "server_auth_token",
     "PUSH_APP_PACKAGE_NAME": "app_package_name", "PUSH_IOS_TOKEN_FORMAT": "ios_token_format",
     "PUSH_PROJECT_ID": "project_id", "PUSH_SERVICE_ACCOUNT_FILE": "service_account_file"})

try:
    logging.warning(f"Push configured for app {push_config.get('PUSH_APP_PACKAGE_NAME')} using platform {push_config.get('PUSH_PROVIDER')} with token {push_config.get('PUSH_SERVER_AUTH_TOKEN')[:10]}... of length {len(push_config.get('PUSH_SERVER_AUTH_TOKEN'))}")
except Exception as e:
    logging.exception(e)
    logging.warning("push service not configured, push notifications not supported")

class NotifyInterfaceFactory(object):
    @staticmethod
    def getDefaultNotifyInterface():
        return NotifyInterfaceFactory.getNotifyInterface(push_config.get("PUSH_PROVIDER"))

    @staticmethod
    def getNotifyInterface(pushProvider):
        module_name = "emission.net.ext_service.push.notify_interface_impl.%s" % pushProvider
        logging.debug("module_name = %s" % module_name)
        module = importlib.import_module(module_name)
        logging.debug("module = %s" % module)
        interface_obj_fn = getattr(module, "get_interface")
        logging.debug("interface_obj_fn = %s" % interface_obj_fn)
        interface_obj = interface_obj_fn(push_config)
        logging.debug("interface_obj = %s" % interface_obj)
        return interface_obj

class NotifyInterface(object):
    def get_and_invalidate_entries(self):
        pass

    def send_visible_notification(self, token_map, title, message, json_data, dev=False):
        pass

    def send_silent_notification(self, token_map, title, message, json_data, dev=False):
        pass

    def display_response(self, response):
        pass

