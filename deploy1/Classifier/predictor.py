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
import logging

# As XGBoost model only allows numerical results, so here's a dict to get
# human-readable categories back:
ind2category = {0: 'DELETION OF INTEREST',
                1: 'RETURNED CHECK',
                2: 'BILL',
                3: 'POLICY CHANGE',
                4: 'CANCELLATION NOTICE',
                5: 'DECLARATION',
                6: 'CHANGE ENDORSEMENT',
                7: 'NON-RENEWAL NOTICE',
                8: 'BINDER',
                9: 'REINSTATEMENT NOTICE',
                10: 'EXPIRATION NOTICE',
                11: 'INTENT TO CANCEL NOTICE',
                12: 'APPLICATION',
                13: 'BILL BINDER'}


# Logger format and location
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='log/testApp.log',
                    filemode='w')

#Define the path
# prefix = 'ml/'
# model_path = os.path.join(prefix, 'model')
model_path = 'ml/model/'
logging.info("Model Path" + str(model_path))

# Load the tfidfVectorizer transformer
tfidf = load(os.path.join(model_path, 'tfidfVectorizer.pkl'))
print(f"loaded tfidf:\t{type(tfidf)}")
logging.info(f"loaded tfidf:\t{type(tfidf)}")

# Load the model components
classifierNB = load(os.path.join(model_path, 'ComplementNaiveBayes0.pkl'))
print(f"loaded classifierNB:\t{type(classifierNB)}")
logging.info(f"loaded classifierNB:\t{type(classifierNB)}")

classifierGB = load(os.path.join(model_path, 'GradientBoostBest.pkl.pkl'))
print(f"loaded classifierGB:\t{type(classifierGB)}")
logging.info(f"loaded classifierGB:\t{type(classifierGB)}")

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

    if useModel == 'NaiveBayes':
        predictions = classifierNB.predict(X)
    elif useModel == 'XGBoosted':
        predictions = [ind2category(p) for p in classifierGB.predict(X)]
    else:
        predictions = ["400 Bad Request"]
    print("categories:\n", predictions)

    # Transform predictions to JSON
    result = {
        'model': f"{useModel}",
        'output': list(predictions)
        }

    resultjson = json.dumps(result)
    return flask.Response(response=resultjson, status=200,
                          mimetype='application/json')

if __name__ == '__main__':    
    # listen on all IPs 
    app.run(host='0.0.0.0', debug=False, port=5000)
