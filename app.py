import pickle
import pandas as pd
import numpy as np
from src.pipelines.prediction_pipeline import CustomData,PredictPipeline
from flask import Flask,request,render_template,jsonify,redirect,url_for,session
import warnings 
warnings.filterwarnings('ignore')

## 1. import ridge regressor model and standard scaler pickle





app = Flask(__name__)
app.secret_key = "forest-fire"


## Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST']) # type: ignore
def predict_datapoint():
    if request.method=='POST':
        data=CustomData(
        Temperature=float(request.form.get('Temperature', 0)),
        RH=float(request.form.get('RH', 0)), 
        Ws=float(request.form.get('Ws', 0)),
        Rain=float(request.form.get('Rain', 0)),
        FFMC=float(request.form.get('FFMC', 0)),
        DMC=float(request.form.get('DMC', 0)),
        ISI=float(request.form.get('ISI', 0)),
        Classes=int(request.form.get('Classes', 0)),
        Region=int(request.form.get('Region', 0))
        )

        pred_df=data.get_data_as_dataframe()
        print(pred_df)

        prediction_pipeline=PredictPipeline()
        result=prediction_pipeline.predict(pred_df)

        session['result'] = round(result[0],2)

            
        return redirect(url_for('predict_datapoint'))
    else:
        result = session.pop('result', None)
        return render_template('home.html', result=result)
    
if __name__=="__main__":
    app.run(host="0.0.0.0")