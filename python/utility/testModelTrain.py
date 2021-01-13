from ModelTrain import *
import numpy as np
import pandas as pd
import pickle


def testPrettifyGridSearchCVResults():
    with open('GridSearchSearchObjEx.pkl', 'rb') as inFile:
        search = pickle.load(inFile)

    
