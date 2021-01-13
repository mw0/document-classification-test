import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV


def dumpAttr(obj):
   for attr in dir(obj):
       if hasattr(obj, attr):
           print(f"obj.{attr}\t= {getattr(obj, attr)}")


def prettifyGridSearchCVResults(search):

    dropCols = ['mean_fit_time', 'std_fit_time', 'mean_score_time',
                'std_score_time', 'split0_test_score', 'split1_test_score',
                'split2_test_score', 'split3_test_score', 'split4_test_score']
    cvResults = pd.DataFrame.from_dict(search.cv_results_, orient='index')
    cvResults.drop(dropCols, inplace=True)
    cvResults = cvResults.T
    cvResults.sort_values(by=['rank_test_score'], ascending=True,
                          inplace=True)
    return cvResults.set_index('rank_test_score')

