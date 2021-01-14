# -*- coding: utf-8 -*-
"""
Created Wed Jan 13 04:15:02 PM PST 2021
@author: Mark Wilber
based upon [example from Naresh Reddy](https://medium.com/analytics-vidhya/deploy-your-own-model-with-aws-sagemaker-55b4234be4a)
"""

# import sys
import os
import json
from joblib import load
import flask
import boto3
# import time
# import pyarrow
# from pyarrow import feather
# from boto3.s3.connection import S3Connection
# from botocore.exceptions import ClientError
# import pickle
# import modin.pandas as pd
# import pandas as pd
import logging

# print(sys.path)

# Logger format and location
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='log/testApp.log',
                    filemode='w')

#Define the path
prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')
logging.info("Model Path" + str(model_path))

# Load the model components
tfidf = load(os.path.join(model_path, 'tfidfVectorizer.pkl'))
print(f"loaded tfidf.\n{type(tfidf)}")
logging.info(f"loaded tfidf.\n{type(tfidf)}")

# Load the model components
classifier = load(os.path.join(model_path, 'ComplementNaiveBayes0.pkl'))
print(f"loaded classifier.\n{type(classifier)}")
logging.info(f"loaded classifier.\n{type(classifier)}")

# The flask app for serving predictions
app = flask.Flask(__name__)
@app.route('/ping', methods=['GET'])
def ping():
    # Check if the classifier was loaded correctly
    try:
        #classifier
        status = 200
        logging.info("Status : 200")
    except:
        status = 400
    return flask.Response(response= json.dumps(' '),
                          status=status, mimetype='application/json' )

@app.route('/invocations', methods=['POST'])
def invocations():
    # Get input JSON data and convert it to a DF
    input_json = flask.request.get_json()

    useModel = input_json['model']
    logging.info(f"useModel: {useModel}.")

    strings = input_json['strings']
    print(strings[0])

    # tokens = []
    # for doc in strings:
        # tokens.append(doc.split())

    X = tfidf.transform(strings)
    predictions = classifier.predict(X)
    print("categories:\n", predictions)
    logging.info(f"useModel: {useModel}.")

    # Transform predictions to JSON
    result = {
        'output': list(predictions)
        }

    resultjson = json.dumps(result)
    return flask.Response(response=resultjson, status=200,
                          mimetype='application/json')

if __name__ == '__main__':    
    # listen on all IPs 
    app.run(host='0.0.0.0', debug=False, port=5000)
