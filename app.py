import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from flask import Flask,request,render_template,jsonify,redirect,url_for,session
import warnings 
warnings.filterwarnings('ignore')

## 1. import ridge regressor model and standard scaler pickle

ridge_model=pickle.load(open('models/ridge_model.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))



app = Flask(__name__)
app.secret_key = "super_secret_key_change_me"


## Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST']) # type: ignore
def predict_datapoint():
    if request.method=='POST':
        Temperature=float(request.form.get('Temperature', 0)) # type: ignore
        RH= float(request.form.get('RH', 0)) # pyright: ignore[reportArgumentType]
        Ws= float(request.form.get('Ws', 0))
        Rain= float(request.form.get('Rain', 0))
        FFMC= float(request.form.get('FFMC', 0))
        DMC= float(request.form.get('DMC', 0))
        ISI= float(request.form.get('ISI', 0))
        Classes= int(request.form.get('Classes', 0))
        Region= float(request.form.get('Region', 0))

        new_data_scaled=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data_scaled)

        session['result'] = float(result[0])

        return redirect(url_for('predict_datapoint'))
    else:
        result = session.pop('result', None)
        return render_template('home.html',result=result)

if __name__=="__main__":
    app.run(host="0.0.0.0")