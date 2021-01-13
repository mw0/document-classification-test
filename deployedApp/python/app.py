# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import pickle
from io import StringIO
import sys
import signal
import traceback
import boto3
from joblib import load
import json

import flask

import pandas as pd
import numpy as np

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

BUCKET_NAME = 'blackknight'
MODEL_FILE_NAME = 'GradientBoostBest.pkl'
TFIDF_FILE_NAME = 'tfidfVectorizer.pkl'

S3 = boto3.client('s3', region_name='us-west-1')


def prediction(data, model, clf):
    """ Calculate test dataset accuracy.
    Args:
        data (DataFrame): Contains labels and sentences
        model (dict): Contains word embeddings into vectors
        clf (object): A classification model
    Returns:
        test_acc (scalar): Test accuracy value
    """

    prediction = []
    try:
        data = data.split()
    except:
        print('Could not split the data: {}'.format(data))
        return prediction


    features = []
    for index_word in data:
        try:
            # If a word encoding does not exist
            # skip it
            features.append(model[index_word])
        except:
            continue
    if features:
        mean_features = np.mean(np.array(features), axis=0)
        prediction = clf.predict([mean_features])


    return prediction

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the
# input data.

class ScoringService(object):
    model = None                # Where we keep the model when it's loaded

    @classmethod
    def get_tfidfVectorizer(self, key):
        """
        Get the tfidfVectorizer object for this instance, loading it if it's not
        already loaded.
        """

        if self.tfidf == None:
            response = S3.get_object(Bucket=BUCKET_NAME, Key=key)
            tfidf_str = response['Body'].read()
            print("tfidf_str:")
            time.sleep(1)
            print(tfidf_str)

            self.tfidf = load(tfidf_str)

            # with open(os.path.join(model_path, 'text-model.pkl'), 'rb') as inp:
            #     self.model = pickle.load(inp)

        return self.model

    @classmethod
    def get_model(self, key):
        """
        Get the model object for this instance, loading it if it's not
        already loaded.
        """

        if self.model == None:
            response = S3.get_object(Bucket=BUCKET_NAME, Key=key)
            model_str = response['Body'].read()
            print("model_str:")
            time.sleep(1)
            print(model_str)

            self.model = load(model_str)

            # with open(os.path.join(model_path, 'text-model.pkl'), 'rb') as inp:
            #     self.model = pickle.load(inp)

        return self.model

    @classmethod
    def predict(self, input):
        """
        For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the
            predictions. There will be one prediction per row in the dataframe.
        """
        tfidf = self.get_tfidfVectorizer(TFIDF_FILE_NAME)
        model = self.get_model(MODEL_FILE_NAME)

        predicted_label = []
        try:
            input['sentence'] = input['sentence'].str.replace('[^\w\s]','')
            input['sentence'] = input['sentence'].\
                apply(lambda x: " ".join(x.lower() for x in x.split()))
        except:
            print('Could not process the text: {}'.format(input))
            return predicted_label

        for a in range(len(input)):
            prediction_one_sample = prediction(input['sentence'][a],
                                               model[1], model[0])
            predicted_label.append(prediction_one_sample)
        return predicted_label

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """
    Determine if the container is working and healthy. In this sample
    container, we declare it healthy if we can load the model successfully.
    """

    # You can insert a health check here
    health = ScoringService.get_model() is not None

    status = 200 if health else 404
    return flask.Response(response='\n', status=status,
                          mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """
    Do an inference on a single batch of data. In this sample server, we take
    data as CSV, convert it to a pandas data frame for internal use and then
    convert the predictions back to CSV (which really just means one
    prediction per line, since there's a single column.
    """

    data = None

    # Convert from CSV to pandas
    if flask.request.content_type == 'text/csv':
        data = flask.request.data.decode('utf-8')
        s = StringIO(data)
        data = pd.read_csv(s,
                           sep=",",
                           quotechar='"',
                           names=['sentence'],
                           skipinitialspace=True,
                           encoding='utf-8',
                           header=None)
    else:
        return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')

    print('Invoked with {} records'.format(data.shape[0]))

    # Do the prediction
    predictions = ScoringService.predict(data)
    
    # Convert from numpy back to CSV
    out = StringIO()
    if (predictions != [] and predictions != [[]]):
        pd.DataFrame({'predicted_label':predictions[0]}).\
            to_csv(out, header=False, index=False)
    else:
        out.write("{\"predicted_label\":\"Unable to predict\"}")

    result = out.getvalue()

    return flask.Response(response=result, status=200, mimetype='text/csv')
