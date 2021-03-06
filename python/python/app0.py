from flask import Flask
from flask import request
from flask import json
import boto3
from joblib import load
import time
# import pickle

BUCKET_NAME = 'blackknight'
MODEL_FILE_NAME = 'GradientBoostBest.pkl'

app = Flask(__name__)

S3 = boto3.client('s3', region_name='us-west-1')


def memoize(f):
    memo = {}

    def helper(x):
        if x not in memo:
            memo[x] = f(x)
        return memo[x]

    return helper


@app.route('/', methods=['POST'])
def index():
    body_dict = request.get_json(silent=True)
    data = body_dict['data']
    time.sleep(2)
    print("data:\n", data)

    prediction = predict(data)

    result = {'prediction': prediction}
    return json.dumps(result)


# @memoize
def load_model(key):
    response = S3.get_object(Bucket=BUCKET_NAME, Key=key)
    model_str = response['Body'].read()
    time.sleep(2)
    print(f"model_str: {model_str}")

    # model = pickle.loads(model_str)
    model = load(model_str)
    time.sleep(2)
    print(type(model))

    return model


def predict(data):
    model = load_model(MODEL_FILE_NAME)

    return model.predict(data).tolist()


if __name__ == '__main__':
    # listen on all IPs
    app.run(host='0.0.0.0')
