#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Randomly selects lines from source file, from which text strings (hashed
# tokens) are extracted, and inserts them into a payload submitted to endpont.
# Categorical responses are compared with those in source, to construct a
# confusion matrix.

import sys
import os
import boto3
import json
import random
import linecache
from argparse import ArgumentParser
from sklearn.metrics import confusion_matrix, classification_report

ACCESS_KEY_ID = os.environ['ACCESS_KEY_ID']
SECRET_ACCESS_KEY = os.environ['SECRET_ACCESS_KEY']
SESSION_TOKEN = None

categories = ['DELETION OF INTEREST', 'RETURNED CHECK', 'BILL',
              'POLICY CHANGE', 'CANCELLATION NOTICE', 'DECLARATION',
              'CHANGE ENDORSEMENT', 'NON-RENEWAL NOTICE', 'BINDER',
              'REINSTATEMENT NOTICE', 'EXPIRATION NOTICE',
              'INTENT TO CANCEL NOTICE', 'APPLICATION', 'BILL BINDER']

def testAPI(endpointName, datafileName=None, modelName=None, sampleSize=None):

    if int(sampleSize) > 1000:
        raise ValueError("API barfs on batch sizes > 1000 (depending upon "
                         "string lengths).")

    runtime = boto3.Session().client('sagemaker-runtime',
                                     region_name='us-east-1',
                                     aws_access_key_id=ACCESS_KEY_ID,
                                     aws_secret_access_key=SECRET_ACCESS_KEY,
                                     aws_session_token=SESSION_TOKEN)


    # payload = {"model": f"{modelName}",
    #            "words": (["135307dba198 b73e657498f2 26f7353edc2e "
    #                       "cd50f04925dd d38820625542"])}

    with open(datafileName, 'rt') as inFile:
        for length, l in enumerate(inFile):
            pass
        length += 1
    # print(length, sampleSize)
    inds = random.sample(range(length), int(sampleSize))
    print(inds)

    cats = []
    lines = []
    for i in inds:
        line = linecache.getline(datafileName, i).rstrip()
        c, w = line.split(',')
        cats.append(c)
        lines.append(w)

    payload = {"model": modelName,
               "words": lines}

    JSONpayload = json.dumps(payload)
    # print(JSONpayload)

    response = runtime.invoke_endpoint(EndpointName=endpointName,
                                       ContentType='application/json',
                                       Body=JSONpayload)
    predictions = json.loads(response['Body'].read().decode())
    yTe = predictions['prediction']

    print(confusion_matrix(cats, yTe))
    print(classification_report(yTe, cats, target_names=categories))


if __name__ == "__main__":
    parser = ArgumentParser(description=('Test remote API with random lines '
                                         'from specified data file.'
                                         '(Note: use sample sizes ≤ 1000.)'))
    parser.add_argument('-e', action='store', dest='endpointName',
                        default='blackknightapp3-endpoint')
    parser.add_argument('-d', action='store', dest='datafileName',
                        default='../data/shuffled-full-set-hashed.csv')
    parser.add_argument('-m', action='store', dest='modelName',
                        default='NaiveBayes')
    parser.add_argument('-s', action='store', dest='sampleSize',
                        default='400')

    results = parser.parse_args()
    testAPI(results.endpointName, datafileName=results.datafileName,
            modelName=results.modelName, sampleSize=results.sampleSize)

