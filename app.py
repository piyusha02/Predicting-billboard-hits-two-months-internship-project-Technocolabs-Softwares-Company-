# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 17:54:41 2021

@author: HP
"""
import numpy as np
from flask import Flask, request, render_template
import pickle

app=Flask(__name__)
pickle_in = open('classifier.pkl','rb')
classifier = pickle.load(pickle_in)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/Predict', methods=['POST'])
def Predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    my_prediction = classifier.predict(final_features)
    return render_template('home.html', prediction = my_prediction)

if __name__=='__main__':    
    app.run()