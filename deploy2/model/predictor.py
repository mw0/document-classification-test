# -*- coding: utf-8 -*-
#!/usr/bin/python3

from pathlib import Path
import os
import json
import io
import flask
from joblib import load

DEBUG = False

modelPath = Path(os.environ['MODEL_PATH'])

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

class ScoringService(object):

    @classmethod
    def __init__(cls):
        
        cls.tfidf = None
        cls.classifierNB = None
        # cls.classifierGB = None

        tfidfPath = modelPath / 'tfidfVectorizer.pkl'
        if DEBUG:
            print(f"tfidf path: {tfidfPath}")
        try:
            with tfidfPath.open('rb') as f:
                cls.tfidf = load(f)
        except OSError() as err:
            print(f"{err}\t{err.args}\t{err.filename}")
        finally:
            if DEBUG:
                print(f"type(cls.tfidf): {type(cls.tfidf)}")

        NaiveBayesPath = modelPath / 'ComplementNaiveBayes0.pkl'
        if DEBUG:
            print(f"NaiveBayesPath: {NaiveBayesPath}")
        try:
            with NaiveBayesPath.open('rb') as f:
                cls.classifierNB = load(f)
        except OSError() as err:
            print(f"{err}\t{err.args}\t{err.filename}")
        finally:
            if DEBUG:
                print(f"type(cls.classifierNB): {type(cls.classifierNB)}")

        # XGBoost requires a GPU, since trained with one; so using
        # best Random Forest model instead. (It's almost as good!)
        RandomForestPath = modelPath / 'RandomForest1.pkl'
        if DEBUG:
            print(f"RandomForestPath: {RandomForestPath}")
        try:
            with RandomForestPath.open('rb') as f:
                cls.classifierRF = load(f)
        except OSError() as err:
            print(f"{err}\t{err.args}\t{err.filename}")
        finally:
            if DEBUG:
                print(f"type(cls.classifierRF): {type(cls.classifierRF)}")

        print("Done with__init__().")


    @classmethod
    def predict(cls, modelName, stringList):
        """For the input, do the predictions and return them.

        Args:
        input (a pandas dataframe): The data on which to do the
        predictions. There will be one prediction per row in the dataframe
        """

        if DEBUG:
            print("You hit get_predict()!")

        X = cls.tfidf.transform(stringList)

        if modelName == 'NaiveBayes':
            predictions = cls.classifierNB.predict(X)
        elif modelName == 'RandomForest':
            # predictions = [ind2category(p)
            #                for p in cls.classifierRF.predict(X)]
            predictions = cls.classifierRF.predict(X)
        else:
            badRequestStr = f"Bad Request (modelName: {modelName})"
            return flask.Response(response=badRequestStr, status=400,
                                            mimetype='text/plain')

        if DEBUG:
            print("categories:\n", predictions)

        return predictions

    @classmethod
    def health(cls):
        if DEBUG:
            print("You hit health().")
        health = ((cls.tfidf is not None) and (cls.classifierRF is not None))
        return health


# The flask app for serving predictions
app = flask.Flask(__name__)
svc = ScoringService()


@app.route('/ping', methods=['GET'])
def ping():
    """
    Determine if the container is working and healthy. In this sample
    container, we declare it healthy if we can load the model successfully.
    """

    # You can insert a health check here
    if DEBUG:
        print("You hit ping!")

    health = svc.health()
    status = 200 if health else 404
    if DEBUG:
        print(f"health: {health}, status: {status}")

    return flask.Response(response=json.dumps('\n'), status=status,
                          mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def invocations():
    """
    Get JSON input and extract modelName and stringList
    """
    if DEBUG:
        print("You hit invocations!")
    input_json = flask.request.get_json()

    modelName = input_json['model']
    if DEBUG:
        print(f"modelName: {modelName}.")

    stringList = input_json['strings']
    if DEBUG:
        print(stringList[0])

    if DEBUG:
        print(f'Invoked with {len(stringList)} records')

    # Do the prediction
    predictions = svc.predict(modelName, stringList)

    # Transform predictions to JSON
    result = {
        'model': f"{modelName}",
        'output': list(predictions)
        }

    resultjson = json.dumps(result)
    return flask.Response(response=resultjson, status=200,
                          mimetype='application/json')


if __name__ == '__main__':    
    # listen on all IPs 
    app.run(host='0.0.0.0', debug=False, port=5000)
