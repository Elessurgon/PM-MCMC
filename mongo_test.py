import pandas as pd
from pymongo import MongoClient

import os
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
import gc
import numpy as np

import seaborn as sns
import patsy as pt
import pymc3 as pm

import sys
import re


def get_data():
    CONNECTION_STRING = "mongodb://localhost:27017/MCMC"
    client = MongoClient(CONNECTION_STRING)
    database = client["MCMC"]
    sales = database["sales"]
    calender = database["calendar"]
    return pd.DataFrame(list(sales.find())), pd.DataFrame(list(calender.find()))
