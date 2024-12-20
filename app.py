from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from src.exception import CustomException
import sys

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 


@app.route('/Predictdata',methods=['GET','POST'])
def predict_datapoint():

    try:

        if request.method=='GET':
          return render_template('home.html')
       
     

        else:
            data=CustomData(
            ApplicantIncome= request.form.get('ApplicantIncome'),
            CoapplicantIncome=float(request.form.get('CoapplicantIncome')),
            Loan_Amount_Term=float(request.form.get('Loan_Amount_Term')),
            Gender=request.form.get('Gender'),
            Married =request.form.get('Married'),
            Dependents=request.form.get('Dependents'),
            Education=request.form.get('Education'),
            Self_Employed=request.form.get('Self_Employed'),
            Credit_History= float(request.form.get('Credit_History')),
            Property_Area= request.form.get('Property_Area'))
            
            pred_df=data.get_data_as_data_frame()
            print("pred_df=",pred_df)
            print("pred_df=",pred_df.shape)
            # print("Before Prediction")

            predict_pipeline=PredictPipeline()
            print("Mid Prediction")
            results=predict_pipeline.predict(pred_df)
            print("after Prediction")
            print("predict result",results)
            return render_template('home.html',results=results[0])
    except Exception as e:
        raise CustomException(e, sys)
# @app.errorhandler(500)
# def handle_error(error):
#     return "Something went wrong", 500

    

if __name__=="__main__":
    app.run(debug=False, port=5002)  # remove debug == true