# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
# import pickle
# from io import StringIO
import sys
import signal
import traceback
import boto3
from joblib import load
import json

import flask

# import pandas as pd
# import numpy as np

# prefix = '/opt/ml/'
# prefix = './model'
# model_path = os.path.join(prefix, 'model')

BUCKET_NAME = 'blackknight'
MODEL_FILE_NAME = 'GradientBoostBest.pkl'
TFIDF_FILE_NAME = 'tfidfVectorizer.pkl'

print("b")
S3 = boto3.client('s3', region_name='us-west-1')


def prediction(data, vectorizer, classifier):
    """ Calculate test dataset accuracy.
    Args:
        data		list(type=str): each string having hashed tokens, 
                          representing a document
        vectorizer	tfidfVectorizer: object for transforming strings
        classifier	sklearn model: classifier for strings

    Returns:
        docClass	str: the selected class
    """

    X = None
    try:
        X = vectorizer.transform(data)
    except:
        print(f'Could not vectorize:\n{data[0]} ...')


    if X:
        prediction = classifier.predict(X)

    return prediction

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the
# input data.

class ScoringService(object):
    vectorizer = None		# Where we keep vectorizer
    classifier = None		# Where we keep classifier

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

        return self.vectorizer

    @classmethod
    def get_classifier(self, key):
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
        model = self.get_classifier(MODEL_FILE_NAME)

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
    health = ((ScoringService.get_vectorizer() is not None) and
              (ScoringService.get_classifier() is not None))

    status = 200 if health else 404
    return flask.Response(response='\n', status=status,
                          mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """
    Do an inference on a single batch of data. In this sample server, we take
    data as list of strings of tokens. If only one string, create a list of
    length 1.
    """

    data = None

    # 
    if flask.request.content_type == 'application/json':
        data = flask.jsonify(flask.request.json)
        # data = flask.request.data.decode('utf-8')
        # data = json.loads(request.get_data().decode("utf-8"))
        print("data:\n", data)
    else:
        return flask.Response(response=('This predictor only supports '
                                        'JSON data'),
                              status=415, mimetype='text/plain')

    print(f'Invoked with {len(data)} records')

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

if __name__ == "__main__":
    print("a")
