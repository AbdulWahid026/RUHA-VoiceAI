import os
import pickle
import librosa
import numpy as np
from flask import Flask, render_template
import numpy

file1 = "ExtraTree_Model.pkl"
file2 = "RandomForsest_Model.pkl"
file3 = "GradientBoosting_Model.pkl"
file4 = "Svm_Model.pkl"
file5 = "DecisionTree_Model.pkl"
file6 = "Knn_Model.pkl"
directory = r"C:\Users\HP\Downloads\i20-2367\Voice Recorder"

recorder = Flask(__name__)


@recorder.route('/')
def index():
    return render_template('voice recorder.html')


def predict():
    os.chdir(directory)
    print(os.listdir())
    extratree_model = pickle.load(open(file1, 'rb'))
    RandomForsest_Model = pickle.load(open(file2, 'rb'))
    GradientBoosting_Model = pickle.load(open(file3, 'rb'))
    Svm_Model = pickle.load(open(file4, 'rb'))
    DecisionTree_Model = pickle.load(open(file5, 'rb'))
    Knn_Model = pickle.load(open(file6, 'rb'))

    mfcc = []
    os.chdir(r"C:\Users\HP\Downloads")
    print(os.getcwd())
    for i in os.listdir():
        if i == 'recording.wav':
            temp1, temp2 = librosa.load(i)
            f = librosa.feature.mfcc(y=temp1, sr=temp2)
            mfcc.append(np.mean(f.T, axis=0))

    prediction1 = Knn_Model.predict(list(mfcc))
    prediction2 = DecisionTree_Model.predict(list(mfcc))
    prediction3 = Svm_Model.predict(list(mfcc))
    prediction4 = GradientBoosting_Model.predict(list(mfcc))
    prediction5 = extratree_model.predict(list(mfcc))
    prediction6 = RandomForsest_Model.predict(list(mfcc))
    return prediction1[0], prediction2[0], prediction3[0], prediction4[0], prediction5[0], prediction6[0]


def find():
    res = np.array(predict())
    res = np.bincount(res).argmax()
    text = ["The person talked about A.C", "The person talked about Bulb or Light",
            "The person talked about Song or Gaana",
            "The person talked about T.V"]

    return text[res]


@recorder.route('/result')
def results():
    return render_template('results.html', t=find())


def det():
    hehe = []
    res = predict()
    text = ["The person talked about A.C", "The person talked about Bulb or Light",
            "The person talked about Song or Gaana",
            "The person talked about T.V"]

    for i in range(0, len(res)):
        if i == 0:
            hehe.append(text[res[i]])
        if i == 1:
            hehe.append(text[res[i]])
        if i == 2:
            hehe.append(text[res[i]])
        if i == 3:
            hehe.append(text[res[i]])
        if i == 4:
            hehe.append(text[res[i]])
        if i == 5:
            hehe.append(text[res[i]])
    return hehe


@recorder.route('/result/details')
def details():
    return render_template('details.html', strings=det())


if __name__ == '_app_':
    recorder.debug = True
    recorder.run(use_debug=True, use_reloader=True)
