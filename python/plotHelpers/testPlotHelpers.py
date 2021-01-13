from plotHelpers import *
from time import sleep
import hashlib
import numpy as np
from numpy.random import RandomState
from collections import OrderedDict
from matplotlib import __version__ as mpVersion


@timeUsage
def sleeper(seconds):
    sleep(seconds)


def testPlotConfusionMatrix():

    print(f"mpVersion: {mpVersion}")

    confusionMatrix = np.array([[220, 12, 58, 3, 17],
                                [7, 330, 15, 22, 5],
                                [41, 3, 406, 8, 21],
                                [41, 72, 36, 308, 16],
                                [6, 11, 8, 19, 441]], dtype='int64')

    print(confusionMatrix)
    xlabels = ['a', 'b', 'c', 'd', 'e']
    ylabels = ['a', 'b', 'c', 'd', 'e']
    titleText = "Example 1"

    plotConfusionMatrix(confusionMatrix, xlabels=xlabels, ylabels=ylabels,
                        titleText=titleText, saveAs='png')

    fileName = 'ConfusionMatrixCountsExample1.png'
    with open(fileName, 'rb') as veriFile:
        datums = veriFile.read()
        actualMD5 = hashlib.md5(datums).hexdigest()

    expectedMD5 = '2db9a809f5dfcc903218b6ae92c99d50'

    assert expectedMD5 == actualMD5

    plotConfusionMatrix(confusionMatrix, xlabels=xlabels, ylabels=ylabels,
                        titleText=titleText, type='recall', saveAs='png')

    fileName = 'ConfusionMatrixRecallExample1.png'
    with open(fileName, 'rb') as veriFile:
        datums = veriFile.read()
        actualMD5 = hashlib.md5(datums).hexdigest()

    expectedMD5 = '068f1f522fbcff839e9d59a8fc212bd8'

    assert expectedMD5 == actualMD5

    plotConfusionMatrix(confusionMatrix, xlabels=xlabels, ylabels=ylabels,
                        titleText=titleText, type='precision', saveAs='png')

    fileName = 'ConfusionMatrixPrecisionExample1.png'
    with open(fileName, 'rb') as veriFile:
        datums = veriFile.read()
        actualMD5 = hashlib.md5(datums).hexdigest()

    expectedMD5 = '70e66d748ac073f1f442c29465b1d7af'

    assert expectedMD5 == actualMD5


def testDetailedHistogram():

    randState = RandomState(20)
    values = randState.randint(0, 101, size=1000)
    titleText = "Example 1"

    detailedHistogram(values, xlabel='values', ylabel='freqs',
                      titleText=titleText, saveAs='png')

    fileName = 'DetailedHistExample1.png'
    with open(fileName, 'rb') as veriFile:
        datums = veriFile.read()
        actualMD5 = hashlib.md5(datums).hexdigest()

    expectedMD5 = '7f4dc4a02dcdcc6830beffdf5a9c0a94'

    assert expectedMD5 == actualMD5


def testPlotValueCounts():

    classes = ['Ugly']*14 + ['Loathsome']*15 + ['Weensy']*16 + \
        ['Stanky']*22 + ['Icky']*23 + ['Bad']*28 + ['Good']*31 + ['Smelly']*41
    values = ['14']*14 + ['15']*15 + ['16']*16 + ['22']*22 + ['23']*23 + \
        ['28']*28 + ['31']*31 + ['41']*41
    values = [int(v) for v in values]

    randState = RandomState(20)
    randState.shuffle(values)
    randState = RandomState(20)
    randState.shuffle(classes)

    df = pd.DataFrame({'value': values, 'class': classes})
    print(df.head(10))
    print(df['class'].value_counts())

    titleText = "Example 1"

    plotValueCounts(df, 'class', titleText=titleText, saveAs='png')

    fileName = 'classFrequenciesExample1.png'
    with open(fileName, 'rb') as veriFile:
        datums = veriFile.read()
        actualMD5 = hashlib.md5(datums).hexdigest()

    expectedMD5 = '80a075938e1a5318be3beafc81a62a29'

    assert expectedMD5 == actualMD5


def testSortClassificationReport():
    classificationReport = ('                         precision    recall  '
                            'f1-score   support\n\n   DELETION OF INTEREST '
                            '      0.73      0.57      0.64       115\n    '
                            '     RETURNED CHECK       0.90      0.90      '
                            '0.90      9472\n                   BILL       '
                            '0.33      0.18      0.23       144\n          '
                            'POLICY CHANGE       0.86      0.90      0.88  '
                            '    4464\n    CANCELLATION NOTICE       0.85  '
                            '    0.86      0.86      4864\n            '
                            'DECLARATION       0.88      0.79      0.83    '
                            '   444\n     CHANGE ENDORSEMENT       0.47    '
                            '  0.27      0.34       483\n     NON-RENEWAL '
                            'NOTICE       0.93      0.91      0.92      '
                            '2413\n                 BINDER       0.82      '
                            '0.77      0.80       367\n   REINSTATEMENT '
                            'NOTICE       0.76      0.75      0.76       '
                            '114\n      EXPIRATION NOTICE       0.89      '
                            '0.88      0.89       312\nINTENT TO CANCEL '
                            'NOTICE       0.82      0.87      0.85      '
                            '5280\n            APPLICATION       0.93      '
                            '0.94      0.94      2183\n            BILL '
                            'BINDER       0.96      0.88      0.92       '
                            '374\n\n               accuracy                '
                            '           0.87     31029\n              macro'
                            ' avg       0.80      0.75      0.77     31029'
                            '\n           weighted avg       0.87      0.87'
                            '      0.87     31029\n')
    expectedOut = ('                         precision    recall  f1-score  '
                   ' support\n\n         RETURNED CHECK       0.90      0.90'
                   '      0.90      9472\nINTENT TO CANCEL NOTICE       0.82      0.87      0.85      5280\n    CANCELLATION NOTICE       0.85      0.86    '
                   '  0.86      4864\n          POLICY CHANGE       0.86    '
                   '  0.90      0.88      4464\n     NON-RENEWAL NOTICE     '
                   '  0.93      0.91      0.92      2413\n            '
                   'APPLICATION       0.93      0.94      0.94      2183\n  '
                   '   CHANGE ENDORSEMENT       0.47      0.27      0.34    '
                   '   483\n            DECLARATION       0.88      0.79    '
                   '  0.83       444\n            BILL BINDER       0.96    '
                   '  0.88      0.92       374\n                 BINDER     '
                   '  0.82      0.77      0.80       367\n      EXPIRATION '
                   'NOTICE       0.89      0.88      0.89       312\n       '
                   '            BILL       0.33      0.18      0.23       '
                   '144\n   DELETION OF INTEREST       0.73      0.57      '
                   '0.64       115\n   REINSTATEMENT NOTICE       0.76      '
                   '0.75      0.76       114\n\n               accuracy     '
                   '                      0.87     31029\n              '
                   'macro avg       0.80      0.75      0.77     31029\n    '
                   '       weighted avg       0.87      0.87      0.87     '
                   '31029\n')

    actualOut = sortClassificationReport(classificationReport)

    assert actualOut == expectedOut


def testTimeUsage(capsys):
    seconds = 2.15
    sleeper(seconds)
    captured = capsys.readouterr()
    assert captured.out == "Δt:  2.15s.\n"
