# Standard imports
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import map
from builtins import *
from random import randrange
import logging
import copy
from datetime import datetime, timedelta
from dateutil import parser
from pytz import timezone
import math
import numpy as np

def isMillisecs(ts):
  return not (ts < 10 ** 11)

def Is_place_2(place1,place,radius):
    # print(place)
    if calDistance(place1,place)<radius:
        return True
    else:
        return False

def Include_place_2(lst,place,radius):
    # list of tracking points
    count=0
    for pnt in lst:
        count=count+(1 if calDistance(pnt,place)<=radius else 0)
    if count>0:
        return True
    else:
        return False

def travel_date_time(time1,time2):
    travel_time = time2-time1
    return travel_time.seconds

def haversine_numpy(lon1, lat1, lon2=None, lat2=None, coordinates=False):
    """
    Haversine distance using 'numpy'
    Fully vectorized implementation for optimal performance.
    
    Usage options:
    1. haversine_numpy(lon1, lat1, lon2, lat2) - vectorized calculation with arrays
    2. haversine_numpy(point1, point2, coordinates=False) - single points in geojson format [lon, lat]
    3. haversine_numpy(point1, point2, coordinates=True) - single points with .lon/.lat attributes
    
    :return: distance in meters
    """
    earth_radius = 6371000
    
    # Extract coordinates from different input formats
    if lon2 is None:
        # This is the calDistance(point1, point2) format
        point1, point2 = lon1, lat1
        
        if coordinates:
            # Points have .lat and .lon attributes
            try:
                lat1 = np.array([float(point1.lat)])
                lon1 = np.array([float(point1.lon)])
                lat2 = np.array([float(point2.lat)])
                lon2 = np.array([float(point2.lon)])
            except AttributeError:
                raise ValueError("When coordinates=True, points must have .lat and .lon attributes")
        else:
            # Points are in geojson format [lon, lat]
            try:
                lon1 = np.array([float(point1[0])])
                lat1 = np.array([float(point1[1])])
                lon2 = np.array([float(point2[0])])
                lat2 = np.array([float(point2[1])])
            except (IndexError, TypeError):
                raise ValueError("When coordinates=False, points must be [lon, lat] format")
        
        single_point = True
    else:
        # Ensure all inputs are numpy arrays for vectorized operations
        if not isinstance(lon1, np.ndarray):
            lon1 = np.array(lon1)
        if not isinstance(lat1, np.ndarray):
            lat1 = np.array(lat1)
        if not isinstance(lon2, np.ndarray):
            lon2 = np.array(lon2)
        if not isinstance(lat2, np.ndarray):
            lat2 = np.array(lat2)
        
        single_point = False
    
    # Vectorized calculation
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    lon1, lon2 = np.radians(lon1), np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    distances = 2 * earth_radius * np.arcsin(np.sqrt(a))
    
    # Return scalar for single point calculations, array otherwise
    if single_point:
        return float(distances[0])
    else:
        return distances

def calHeading_numpy(lon1, lat1, lon2, lat2):
    """
    Calculate heading angle using numpy with identical results to calHeading.
    Points are expected to be in longitude, latitude form (WGS84)
    :return heading in degrees, from -180 to 180
    """
    # Ensure all inputs are numpy arrays
    if not isinstance(lon1, np.ndarray):
        lon1 = np.array(lon1)
    if not isinstance(lat1, np.ndarray):
        lat1 = np.array(lat1)
    if not isinstance(lon2, np.ndarray):
        lon2 = np.array(lon2)
    if not isinstance(lat2, np.ndarray):
        lat2 = np.array(lat2)
    
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    lon1, lon2 = np.radians(lon1), np.radians(lon2)
    
    y = np.sin(lon2 - lon1) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    
    return np.degrees(np.arctan2(y, x))

def calDistance(point1, point2, coordinates=False):
    """haversine distance

    :param point1: a coordinate in degrees WGS84
    :param point2: another coordinate in degrees WGS84
    :param coordinates: if false, expect a list of coordinates, defaults to False
    :return: distance approximately in meters
    """
    # This function is maintained for backward compatibility
    # Use haversine_numpy instead for new code
    return haversine_numpy(point1, point2, coordinates=coordinates)

def compare_rounded_arrays(arr1, arr2, digits):
    round2n = lambda x: round(x, digits)
    return list(map(round2n, arr1)) == list(map(round2n, arr2))

