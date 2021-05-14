# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 10:30:58 2021

@author: chithra anil
"""


from flask import Flask,request
import pandas as pd
import numpy as np
import pickle

app=Flask(__name__)
pickle_in=open('classifier.pkl','rb')
classifier=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Welcome All"
@app.route('/predict')
def predict_currency_validation():
    variance=request.args.get('variance')
    skewness=request.args.get('skewness')
    curtosis=request.args.get('curtosis')
    entropy=request.args.get('entropy')
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    return "The prediction is"+ str(prediction)

@app.route('/predict_file',methods=['POST'])
def predict_currency_validation_file():
    df_test=pd.read_csv(request.files.get("file"))
    prediction=classifier.predict(df_test)
    return "The predictions are" + str(list(prediction))
    

if __name__=='__main__':
    app.run()