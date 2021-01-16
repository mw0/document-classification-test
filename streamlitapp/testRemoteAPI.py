#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import boto3
import json

ACCESS_KEY_ID = os.environ['ACCESS_KEY_ID']
SECRET_ACCESS_KEY = os.environ['SECRET_ACCESS_KEY']
SESSION_TOKEN = None

def testAPI(endpointName, modelName, docuStrings):

    runtime = boto3.Session().client('sagemaker-runtime',
                                     region_name='us-east-1',
                                     aws_access_key_id=ACCESS_KEY_ID,
                                     aws_secret_access_key=SECRET_ACCESS_KEY,
                                     aws_session_token=SESSION_TOKEN)

    payload = {"model": f"{modelName}",
               "strings": (["135307dba198 b73e657498f2 26f7353edc2e "
                            "cd50f04925dd d38820625542"])}

#     payload = {"model": f"{modelName}",
#                "strings": docuStrings}

    JSONpayload = json.dumps(payload)
    print(JSONpayload)

    response = runtime.invoke_endpoint(EndpointName=endpointName,
                                       ContentType='application/json',
                                       Body=JSONpayload)
    result = json.loads(response['Body'].read().decode())

    return result

if __name__ == "__main__":
    print(f"{sys.argv}")
    endpointName = sys.argv[1]
    modelName = sys.argv[2]
    docuStrings = sys.argv[3]
    print(testAPI(endpointName, modelName, docuStrings))


