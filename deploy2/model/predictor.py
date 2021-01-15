# -*- coding: utf-8 -*-


from pathlib import Path
import os
import json
import io
import flask
from joblib import load
import logging

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

# Logger format and location
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='./testApp.log',
                    filemode='w')

class ScoringService(object):

    @classmethod
    def __init__(cls):
        print(f"modelPath: {modelPath}")
        tfidfPath = modelPath / 'tfidfVectorizer.pkl'
        print(f"tfidf path: {tfidfPath}")
        try:
            f = tfidfPath.open('rb')
            print(f"type(f): {type(f)}")
            cls.tfidf = load(f)
            print(f"type(cls.tfidf): {type(cls.tfidf)}")
            f.close()
            # with tfidfPath.open('rb') as f:
            #     cls.tfidf = load(f)
        except OSError() as err:
            print(f"{err}\t{err.args}\t{err.filename}")

        try:
            with NaiveBayesPath.open('rb') as f:
                cls.classifierNB = load(f)
        except OSError() as err:
            print(f"{err}\t{err.args}\t{err.filename}")

        try:
            with XGBoostPath.open('rb') as f:
                cls.classifierXB = load(f)
        except OSError() as err:
            print(f"{err}\t{err.args}\t{err.filename}")

        print("Done with __init__().")

    @classmethod
    def predict(cls, modelName, stringList):
        """For the input, do the predictions and return them.

        Args:
        input (a pandas dataframe): The data on which to do the
        predictions. There will be one prediction per row in the dataframe
        """

        print("You hit get_predict()!")
        X = cls.tfidf.transform(stringList)

        if modelName == 'NaiveBayes':
            predictions = cls.classifierNB.predict(X)
        elif modelName == 'XGBoosted':
            predictions = [ind2category(p)
                           for p in cls.classifierGB.predict(X)]
        else:
            return flask.Response(response='Bad Request', status=400,
                                  mimetype='text/plain')

        print("categories:\n", predictions)

        return predictions

    @classmethod
    def health(cls):
        print("Checking health!")
        dfidf = cls.tfidf()
        print(f"tfidf: {tfidf is not None}")
        classifierNB = cls.classifierNB()
        print(f"classifierNB: {classifierNB is not None}")
        classifierGB = cls.classifierGB()
        print(f"classifierGB: {classifierGB is not None}")
        health = ((tfidf is not None) and
                  (classifierNB is not None) and
                  (classifierGB is not None))
        print(f"Healty? {health}")

        return health


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route('/ping', methods=['GET'])
def ping():
    """
    Determine if the container is working and healthy. In this sample
    container, we declare it healthy if we can load the model successfully.
    """

    # You can insert a health check here
    print("You hit ping!")
    ss = ScoringService()
    # health = ((ScoringService.tfidf is not None)
    #           and (ScoringService.get_classifierNB() is not None)
    #           and (ScoringService.get_classifierGB() is not None))

    health = ss.health()
    print(f"health: {health}")

    status = 200 if health else 404
    print(f"status: {status}")
    return flask.Response(response='\n', status=status,
                          mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def invocations():
    """
    Get JSON input and extract modelName and stringList
    """

    ss = ScoringService()
    input_json = flask.request.get_json()

    modelName = input_json['model']
    logging.info(f"modelName: {modelName}.")

    stringList = input_json['strings']
    print(stringList[0])

    print(f'Invoked with {len(stringList)} records')

    # Do the prediction
    predictions = ss.predict(modelName, stringList)

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
