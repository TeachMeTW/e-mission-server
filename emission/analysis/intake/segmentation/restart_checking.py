from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import *
import pandas as pd
import numpy as np
import logging
import emission.core.wrapper.transition as ecwt
import emission.storage.timeseries.timequery as estt


def ongoing_motion_in_loc_df(loc_df, motion_df):
    """
    Return a Series of booleans that indicates, for each row in loc_df,
    whether there was any background/motion_activity between the previous
    row and the current row.
    """
    if motion_df.empty:
        return pd.Series(False, index=loc_df.index)
    
    motion_timestamps = motion_df['ts'].values
    motion_idxs = np.searchsorted(loc_df['ts'].values, motion_timestamps)
    motion_idxs = motion_idxs[motion_idxs < len(loc_df)]
    logging.debug(f'motion_timestamps = {motion_timestamps}, motion_idxs = {motion_idxs}')
    ongoing_motion = pd.Series(False, index=loc_df.index)
    ongoing_motion[motion_idxs] = True
    return ongoing_motion


def tracking_restarted_in_loc_df(loc_df, transition_df):
    """
    Return a Series of booleans that indicates, for each row in loc_df,
    whether there were any tracking restarts between the previous
    row and the current row.

    A tracking restart is defined as one of:
     - a STOP_TRACKING transition
     - a BOOTED transition (Android)
     - a VISIT_ENDED transition with curr_state WAITING_FOR_TRIP_START (iOS).
    """
    if transition_df.empty:
        return pd.Series(False, index=loc_df.index)
    
    restart_transitions_df = transition_df[
        (transition_df.transition == ecwt.TransitionType.STOP_TRACKING.value) |
        # Android
        (transition_df.transition == ecwt.TransitionType.BOOTED.value) |
        # iOS
        ((transition_df.curr_state == ecwt.State.WAITING_FOR_TRIP_START.value) &
         (transition_df.transition == ecwt.TransitionType.VISIT_ENDED.value))
    ]
    restart_timestamps = restart_transitions_df['ts'].values
    restart_idxs = np.searchsorted(loc_df['ts'].values, restart_timestamps)
    restart_idxs = restart_idxs[restart_idxs < len(loc_df)]
    logging.debug(f'restart_timestamps = {restart_timestamps}, restart_idxs = {restart_idxs}')
    tracking_restarted = pd.Series(False, index=loc_df.index)
    tracking_restarted[restart_idxs] = True
    return tracking_restarted


def is_tracking_restarted_in_range(start_ts, end_ts, transition_df=None):
    """
    Check to see if tracing was restarted between the times specified
    :param start_ts: the start of the time range to check
    :param end_ts: the end of the time range to check
    :param timeseries: the timeseries to use for checking
    :param transition_df: dataframe of transitions to use (if None, will be fetched from timeseries)
    :return:
    """
    transition_df = transition_df[
        (transition_df['ts'] >= start_ts) & (transition_df['ts'] <= end_ts)
    ]

    if len(transition_df) == 0:
        logging.debug("In range %s -> %s found no transitions" %
                      (start_ts, end_ts))
        return False
    logging.debug("In range %s -> %s found transitions %s" %
                  (start_ts, end_ts, transition_df[["fmt_time", "curr_state", "transition"]]))
    return _is_tracking_restarted_android(transition_df) or \
           _is_tracking_restarted_ios(transition_df)

def get_ongoing_motion_in_range(start_ts, end_ts, timeseries, motion_list=None):
    if motion_list is not None:
        motion_list = [
            m for m in motion_list if m['data']['ts'] >= start_ts and m['data']['ts'] <= end_ts
        ]
    else:
        tq = estt.TimeQuery(timeType = "data.ts", startTs = start_ts,
                            endTs = end_ts)
        motion_list = list(timeseries.find_entries(["background/motion_activity"], tq))

    logging.debug("Found %s motion_activity entries in range %s -> %s" %
                  (len(motion_list), start_ts, end_ts))
    logging.debug("sample activities are %s" % motion_list[0:5])
    return motion_list

def _is_tracking_restarted_android(transition_df):
    """
    Life is fairly simple on android. There are two main cases that cause tracking
    to be off. Either the phone is turned off, or the user manually turns tracking off
    tracking.
    - If the phone is turned off and turned on again, then we receive a REBOOT transition,
    and we INITIALIZE the FSM.
    - If the tracking is turned off and on manually, we should receive a STOP_TRACKING + START_TRACKING
    transition, followed by an INITIALIZE

    Should we use the special transitions or INITIALIZE? We also call INITIALIZE when we
    change the data collection config, which doesn't necessarily require a break in the chain.
    Let's use the real transitions for now.
    :param transition_df: the set of transitions for this time range
    :return: whether the tracking was restarted in that time range
    """
    restart_events_df = transition_df[(transition_df.transition == ecwt.TransitionType.BOOTED.value) |
                                      (transition_df.transition == ecwt.TransitionType.STOP_TRACKING.value)]
    if len(restart_events_df) > 0:
        logging.debug("On android, found restart events %s" % restart_events_df)
    return len(restart_events_df) > 0

def _is_tracking_restarted_ios(transition_df):
    """
    Unfortunately, life is not as simple on iOS. There is no way to sign up for a
    reboot transition. So we don't know when the phone rebooted. Instead, geofences
    are persisted across reboots. So the cases are:
    - phone runs out of battery in:
        - waiting_for_geofence:
            - we turn on phone in a different location: geofence is triggered,
             we are OUTSIDE geofence, we start tracking, we detect we are not moving,
             we turn off tracking. There doesn't appear to be anything that we can
             use here to distinguish from a spurious trip without reboot, except that
             in a spurious trip, the new location is the same while in this, it is different.
             but that is covered by the distance checks above.

           - we turn on phone in the same location: nothing to distinguish whatsoever.
             Have to let this go
        - ongoing_trip:
           - turn on as part of the same trip - will be initialized with the significant
             location changes API. no way to distinguish between significant and
             standard location changes, so it will just look like a big gap in points
             (greater than the distance filter)
           - turn on at home - barring visit notifications, will have tracking turned off
             until next trip starts and we get a significant location changes API update.
             Need to test how visit tracking interacts with all this

    Experimental results at:
    https://github.com/e-mission/e-mission-server/issues/378

    :param transition_df: the set of transitions for this time range
    :return: whether the tracking was restarted in that time range
    """
    restart_events_df = transition_df[(transition_df.transition == ecwt.TransitionType.STOP_TRACKING.value) |
                                      ((transition_df.curr_state == ecwt.State.WAITING_FOR_TRIP_START.value) &
                                       (transition_df.transition == ecwt.TransitionType.VISIT_ENDED.value))]
    if len(restart_events_df) > 0:
        logging.debug("On iOS, found restart events %s" % restart_events_df)
    return len(restart_events_df) > 0


