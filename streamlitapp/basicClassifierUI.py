#!/usr/bin/python3
# -*- coding: utf-8 -*-

#### Imports

from time import time, asctime, localtime, perf_counter
print(asctime(localtime()))

import os
import json
import re

import sys
import streamlit as st
import pandas
import requests
import boto3

# Radio button descriptors are too long, so shorten them with this dict:
modelNamer = {"Naive Bayes (baseline)": "NaiveBayes", "XGBoosted (optimized)": "XGBoosted"}

ACCESS_KEY_ID = os.environ['ACCESS_KEY_ID']
SECRET_ACCESS_KEY = os.environ['SECRET_ACCESS_KEY']
SESSION_TOKEN = None

endpointName='blackknight-classifier02h-endpoint'
runtime = boto3.Session().client('sagemaker-runtime',
                                 region_name='us-east-1',
                                 aws_access_key_id=ACCESS_KEY_ID,
                                 aws_secret_access_key=SECRET_ACCESS_KEY,
                                 aws_session_token=SESSION_TOKEN)

#### cacheable functions
@st.cache()
def createJSONrequestStr(modelString, docuString):
    """
    modelString	str: the string returned from the radio button
    docuString	str: String from text_area widget that is to be converted to JSON

    string is split using re.split(r'\n+', docuString), and modelString is replaced using
    modelNamer dict.
    """

    modelStr = modelNamer[modelString]
    docStrList = re.split(r'\n+', docuString)
    if len(docStrList) <= 1:
        docStrList = [docStrList]
    requestDict = {'model': modelStr, 'strings': docStrList}

    return json.dumps(requestDict)


# @st.cache()
def invokeEndpoint(endpointName, requestJSON):
    """
    Fetches a response from model endpoint
    """
    response = runtime.invoke_endpoint(EndpointName=endpointName,
                                       ContentType='application/json',
                                       Body=requestJSON)

    return response

#### Start building the app

st.title("Document Classifier UI for Black Knight HeavyWater Problem")
st.info("blame: Mark Wilber")

st.sidebar.title("About")
st.sidebar.info("This is a quick UI for testing classifiers created for the "
                "Problem. Multiple models were trained on TF-IDF features, "
                "with two shown here. The Naive Bayes model with default "
                "settings serves as a baseline, while the XGBoost model was "
                "the result of several hours of grid searching. Other models "
                "performed almost as well as this best XGBoost version, but "
                "model sizes could be as high as 475MB.\n\nWhen you mash on "
                "the 'Send' button a JSON payload is formulated and shipped to"
                " a RESTful API hosted on AWS. It used the model name to "
                "select which version to use for inference. When the returned "
                "payload is returned, this app displays the results.\n\n"
                "For details, see [my fork](https://github.com/mw0/document-"
                "classification-test) of HeavyWater's original github repo.")

# model = st.sidebar.radio("Which model?", ("Naive Bayes (baseline)", "XGBoosted (optimized)"))

showFormattedRequest = False
showFormattedRequest = st.sidebar.checkbox("Show formatted request",
                                           value=False)

model = 'Naive Bayes (baseline)'

inputBox = st.empty()
getButton = st.empty()
resultsBox = st.empty()
JSONbox = st.empty()

defaultDoc = ("4e5019f629a9 54fb196d55ce 0cf4049f1c7c ef4ea2777c02 f8552412da3f 0a9b859f7b89 a31962fbd5f3 2bcce4e05d9d "
              "b61f1af56200 036087ac04f9 6d25574664d2 9cdf4a63deb0 07e7fe209a3b 93c988b67c47 8a3fc46e34c1 b59e343416f7 "
              "e4ed491481ed e7f29d3843e7 87aeadfbc7f2 612cc551a793 9bc65adc033c 6bc122aa4b06 fe3fe35491b4 6faa0d565869 "
              "f0382a00d499 f79da29e041c 3c510ffe475f 543615850429 fc462213a9d4 8f75273e5510 133d46f7ed38 2519927ae3fa "
              "0f1d041f5921 8871e31e57e2 ce68d85c1b08 93790ade6682 9415e522bc59 4357c81e10c1 b208ae1e8232 f79da29e041c "
              "8dedae08a79f 0cbca93be301 35991d8609e2 43af6db29054 6ca2dd348663 6b304aabdcee d38820625542 8e0a28537681 "
              "98c9498f85a3 2f58ef8b979c 932deb3b70cc a3360a4991fa 79aa7fd11cec 133d46f7ed38 cd4c3c5e83cc 0f12fd1c2b99 "
              "31fd3123f41c 33630ee5f812 d9ef68daef4c 0cbca93be301 76b296c8d48d 0cbca93be301 c1a2676df403 f1413affa34b "
              "3d9a3fcc2f1c 86985e33826a b73e657498f2 20127286030f 2d00e7e4d33f f4a65848d21f 6365c4563bd1 6bf9c0cb01b4 "
              "c79b01c5629d e504ee0aaf6d 8b0131ee1005 12654bbe59c7 6c998bcc2f5e b9699ce57810 a2b3b5dae4b9 17b109eb308e "
              "0562c756a2f2 0cbca93be301 8532b14a158d b3d1c1eff9d7 10e45001c2f2 e7f10ad56136 40cb5e209b92 ba963af414e7 "
              "97b6014f9e50 b9699ce57810 6bf9c0cb01b4 3f612d66ae3b d38820625542 28bce73d3237 7c95e780d5ec 7336405c7cfa "
              "0cbca93be301 72bd4a50cf4a 0c490affb30d f3ff20955734 7009efbdb7b1 d38820625542 e1b9e4df3a88 4ad52689d690 "
              "a024d1e04168 c337a85b8ef9 f1393c430fd1 95de2151d27c 586242498a88 8071efb3570c 90e906ce4b90 1015893e384a "
              "d037ca00fa63 22fa1184be26 9e5639a57e81 7860028b1d17 56b98f5aef76 be95012ebf2b 586242498a88 845c5d0ccbf8 "
              "61442b440484 6b343f522f78 d9ef68daef4c ec522bf7f985 ec3406979928 816aed74475e 89cce1c7ef23 43af6db29054 "
              "43af6db29054 8f75273e5510 4ad52689d690 a024d1e04168 1b6d0614f2c7 ef4ba44cdf5f 1015893e384a 1015893e384a "
              "5748149bc6f5 ffe8decfd82e 427028e08976 9bc65adc033c 133d46f7ed38 564aaf0c408b 6defcd633808 de078996c1a5 "
              "5ff8f7117bc9 32150e5d4311 6f6fb5a7797f 48cda5a5171a 288ccf089872 56c2c356d772 3b06427e873d 9bc65adc033c "
              "6ef2ade170d9 d9ef68daef4c e3a330c58136 2ce0277ae4e0 be95012ebf2b 2d00e7e4d33f 0cbca93be301 20d53168dbb6 "
              "86f0841bdf32 40d775b9c777 a921ec62a35d 20feccb59250 0a4e98ab9a8f d38820625542 97b6014f9e50 0562c756a2f2 "
              "8460873735c7 a1bb6b4223d9 f1413affa34b 4ad52689d690 8f7a92cd0ae7 e9be2c2489cd b208ae1e8232 a7d66c72a972 "
              "28bce73d3237 4ad52689d690 8f7a92cd0ae7 c337a85b8ef9 5ff8f7117bc9 c5c52e33fb85 3ddfcec1d334 fe3fe35491b4 "
              "97b6014f9e50 a86f2ba617ec 47439a0d8004 f55be870e57d d19525fb2ce9 61442b440484 63bb06c20a26 7b5fc7b4a3b7 "
              "76b296c8d48d 97b6014f9e50 0cbca93be301 abe7d2dd7c9b c7e5e11eb408 8b363e6fdb6a 878460b4304e 8f75273e5510 "
              "08ef3639a83b 586242498a88 448cca02dae6 80f78de0c6ab c5dcd74b40a9 21a154a4003a 7e8d554779a7 d38820625542 "
              "f49ab97a086c 6c0295f8ccbc f76533ec75cf 66c79e93b047 257ac5b0e629 2fc77c15c39d 1417bc08c137 395a5e8185f8 "
              "7a5e719bafba cbfb3eb99bea f1a45cc91e7a 374f2f4432ff e5e6da4eb92e 1e8b63e5bb3e 5c2db045bc17 56a2e52a4930 "
              "4df55faae9d6 3fb046fb884d 6b304aabdcee 6363498d897e bf15989af17d 037279275d06 c466347b2ae9 99e613bf119e "
              "ae26e97b8730 816aed74475e 5d00ab650ac2 f86490d29db0 9bc65adc033c 8d6768212702 d63be9e66da8 fe33912c5732 "
              "b208ae1e8232 b7ab56536ec4 ee94f34a89db 77a64531e8e5 6b343f522f78 a1bb6b4223d9 0cbca93be301 d911e9441d32 "
              "ffb0084cf8a5 9bc65adc033c 6ce6cc5a3203 d38820625542 c4892276fccc 4b9f066b83a6 6ce6cc5a3203 a3360a4991fa "
              "a9de7423ee4a b7a74dbda8f9 2d00e7e4d33f 8f75273e5510 470aa9b28443 1ab34730c1e0 6b343f522f78 76b296c8d48d "
              "0cbca93be301 cf4fc632eed2 586242498a88 a3008b19a353 b208ae1e8232 b7ab56536ec4 d9ef68daef4c a886e1712428 "
              "db293664bc6d 8f75273e5510 ed5d3a65ee2d e1b9e4df3a88 bf3aa3fc66f6 0283ef044d5c c33b5c3d0449 5ce740893329 "
              "95ef80a0b841 448cca02dae6 c913f5129fe2 35991d8609e2 377a21b394dc d5e335b0e32e d5f4611022c1 76b296c8d48d "
              "0cbca93be301 ec3406979928 aca98b3cfc1b fd3fb0074a05 35991d8609e2 133d46f7ed38 acf0b0d7d8d0 d38820625542 "
              "0cbca93be301 5c2db045bc17 2cde7b991526 6d25574664d2 d575418057fa 9cdf4a63deb0 42a82637cd49 8a3fc46e34c1 "
              "f39c3de43b96 98d0d51b397c 84bf8a94981c b59e343416f7 0f64852d2606 76b296c8d48d 0cbca93be301 0158863fa432 "
              "93790ade6682 22182adbd370 9bc65adc033c a05dfb40dc0a 7fdfab6f73ae b61f1af56200 927b09e4929b f95d0bea231b "
              "036087ac04f9 97b6014f9e50 1669ab06727a 4357c81e10c1 46c88d9303da b4340c07c50c 0562c756a2f2 28439f5ac34b "
              "93c988b67c47 156cc7006ae8 2ef7c27a5df4 816aed74475e 2bcce4e05d9d 288ccf089872 4df20d063468 4ad52689d690 "
              "a024d1e04168 1b6d0614f2c7 4ad52689d690 a024d1e04168 c337a85b8ef9 97b6014f9e50 9a42ead47d1c cc03a8691f3f "
              "0c4ce226d9fe d9ef68daef4c 5948001254b3 4b9d48caa208 7b92f0d3f093 fbb5efbcc5b3 db1df1ae2fec b11c41c83915 "
              "586242498a88 95ef80a0b841 0c4ce226d9fe 132a682c6a03 706c4ce7fa52 99e613bf119e d2fb30f04ab7 b9699ce57810 "
              "89f5f5df8f52 5d7641b096f0 4ad52689d690\n\nad4440ac97a5 8e93a2273a93 c913f5129fe2 bfb030c0e4e2 6ce6cc5a3203"
              " 798fe9915030 42e211f8752a 7eb23b5b9603 f7ae6f8257da 9d634fae0367 2f2548bd374a 25c57acdf805 75df40507e72 "
              "ffe8decfd82e 422068f04236 3e56fed2d392 063a3ef1e75f 8db54a4cb57b 25c57acdf805 e52882a7f2b7 8db54a4cb57b "
              "37ac79620fc6 596fbbd504aa ffe216d9d610 6868362b998e fc96b835cfc3 ffe216d9d610 6868362b998e eca16ee06b98 "
              "25c57acdf805 641356219cbc 422068f04236 5f43e051f9a6 48d657cd9861 fc1955933b8e eca16ee06b98 957b5cf4e65e "
              "422068f04236 fb53275d6678 f56b300cc325 48d657cd9861 6101ed18e42f 586242498a88 48d657cd9861 6b343f522f78 "
              "8db54a4cb57b e7e059c82399 6ca2dd348663 b87f34b0269a bfb030c0e4e2 d38820625542 e943e5e5b779 c8d2304e52cf "
              "fbe267908bc5 2f2548bd374a cbfb3eb99bea 6ce6cc5a3203 d19b1c129f40 5f43e051f9a6 586242498a88 c8f5ad40a683 "
              "4ffb12504ac6 8cb71bb0ee27 66813d53f12a bdba286f728a f7ae6f8257da 938812903b4e 5f43e051f9a6 8cb71bb0ee27 "
              "fbe267908bc5 fbe267908bc5 2f2548bd374a a100eb50abec 2f2548bd374a ad4440ac97a5 cf4fc632eed2 2f2548bd374a "
              "25c57acdf805 422068f04236 d19b1c129f40 a3518ffa104e 5f43e051f9a6 33043bd1c2f4 db108078ec43 5ff8f7117bc9 "
              "8cb71bb0ee27 0e9329e43507 6b3268e10628 e7e059c82399 bfb030c0e4e2 744366456381 e259a56993f4 e3a330c58136 "
              "d671855584fd eeb86a6a04e4 a3518ffa104e d736fc77c54b fbe267908bc5 fbe267908bc5 586242498a88 f7ae6f8257da "
              "a5f8a7c9a886 0c4ce226d9fe 9b88c973ae02 21e314d3afcc 11a897cb0d78 d493c688fb66 8cb71bb0ee27 de9738ee8b24 "
              "7bf4f79c3fd9 6365c4563bd1 9374c105ef84 de9738ee8b24 25c57acdf805 37ac79620fc6 8f7a92cd0ae7 cf4fa36520cb "
              "ad4440ac97a5 eb51798a89e1 8cb71bb0ee27 a100eb50abec f7ae6f8257da f7ae6f8257da 19e9f3592995 586242498a88 "
              "bfb030c0e4e2 37ac79620fc6 8cb71bb0ee27 4ffb12504ac6 10aa76ec946b ffe216d9d610 c24d76b5b80a ba02159e05b1 "
              "033616ad6870 d2cabcd692f6 8f7a92cd0ae7 360e8b28421c 21c66f6b38af a7c177a24cab ffe216d9d610 db108078ec43 "
              "5dc515102cfb ce02bbeeb97f 3b952c633ee4 f7d55eadc647 ad4440ac97a5 033616ad6870 038043bd66da bfb030c0e4e2 "
              "da046a9d8e36 70a81d6fffab b60e6ed0d053 32b5989b13f0 72bd4a50cf4a 1b21cf220a68 ded7b70601fc 292891f020a4 "
              "586242498a88 6f6729c54a07 60439259777c 8cb71bb0ee27 f7ae6f8257da c82f81aceeab fbe267908bc5 2f2548bd374a "
              "ffe216d9d610 ce00eff819b7 25c57acdf805 f7ae6f8257da 0704e636f7b8 ad4440ac97a5 f7ae6f8257da 586242498a88 "
              "21c66f6b38af 8f75273e5510 8cb71bb0ee27 789f72dda0b0 2c129538d383 3b952c633ee4 bfb030c0e4e2 bfb030c0e4e2 "
              "e851bc6d8f3a ad4440ac97a5 d671855584fd 6101ed18e42f d19b1c129f40 33043bd1c2f4 4ffb12504ac6 bfb030c0e4e2 "
              "ad4440ac97a5 25c57acdf805 fbe267908bc5 2f2548bd374a db108078ec43 fbe267908bc5 8cb71bb0ee27 d19b1c129f40 "
              "2f2548bd374a 33043bd1c2f4 37ac79620fc6 8cb71bb0ee27 f7ae6f8257da 586242498a88 6101ed18e42f d2fed0e65ee8 "
              "25c57acdf805 f685674301a1 266dc1fd820c 388c65f7aa33 25c57acdf805 6b343f522f78 5ee06767bc0f 11d62d3598ce "
              "a5f8a7c9a886 774445039259 d2cabcd692f6 a2b31bf99ba2 a56e20cc7707 1c589cc32bc9 534a78d7faac d2cabcd692f6 "
              "da61efdd2b77 caecbc15a560 dec88250479b b9699ce57810 394a95035188 641356219cbc 25c57acdf805 ca870440c1fe "
              "65f888439937 cf6595050f1c 19e9f3592995 ffe216d9d610 029bf0e29a73 4b5f7fa402e5 20a84e403407 b814d9d78802 "
              "2c33750c1d59 a7d9f88a65fa f816f047c0db 63074fe20296 6bff0c8c1185 fe2bf89a5fcb 19e9f3592995 607e30a9689e "
              "25c57acdf805 ad4440ac97a5 23344d9339f0 7460dd389317 fe2bf89a5fcb d91e94eea601 8cb71bb0ee27 f7ae6f8257da "
              "2f2548bd374a fc25f79e6d18 bfb030c0e4e2 d5f4611022c1 9374c105ef84 c8d2304e52cf ad4440ac97a5 5b1787f13fd0 "
              "f95d0bea231b af671fbeb212 25c57acdf805 69e45f35d87e ba02159e05b1 25c57acdf805 f7ae6f8257da 48d657cd9861 "
              "cf4fa36520cb 37ac79620fc6 3a29db408753 ad4440ac97a5 c8f5ad40a683 1fa50b787bd8 9ccf259ca087 7b5fc7b4a3b7 "
              "ad4440ac97a5 26f768da5068 6af770640118")

headingStr = ("Replace these example documents with your own, each separated by at least 1 newline.")
docStrings = inputBox.text_area(headingStr, defaultDoc, max_chars=12000,
                                height=500)

JSONheader = "JSON formatted request"

if getButton.button("Get results!"):
    requestJSON = createJSONrequestStr(model, docStrings)

    if showFormattedRequest:
        JSONout = JSONbox.text_area(JSONheader, requestJSON, max_chars=4000,
                                    height=300)

    print("About to call API.")
    response = invokeEndpoint(endpointName, requestJSON)
    print(f"response: {response}")

    result = json.loads(response['Body'].read().decode())
    verifiedModel = result['model']
    categories = "\n".join(result['output'])

    resultsHeader = "Response from API"
    thing = resultsBox.text_area(resultsHeader,
                                 verifiedModel + ":\n" + categories,
                                 max_chars=600, height=80)

