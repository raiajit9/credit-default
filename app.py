# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
from flask import Flask
from flask import request
from flask import render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model_pickle', 'rb'))

@app.route('/')
def home():
    return render_template('credit.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = ['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE','PAY_0',
                     'PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT1','BILL_AMT2',
                     'BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1',
                     'PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
    
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
        
    if output == 0:
        res_val = "** NOT DEFAULT **"
    else:
        res_val = "DEFAULT"
        

    return render_template('credit.html', prediction_text='CUSTOMER WILL{}'.format(res_val))

if __name__ == "__main__":
    app.run()
