from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)
# app = application
# @app.route('/')
# def index():
#     return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method=='GET':
        return render_template('index.html')
    else:
        data = CustomData(
            pickup_datetime=request.form.get('pickup_datetime'),
            pickup_longitude=request.form.get('pickup_longitude'),
            pickup_latitude=request.form.get('pickup_latitude'),
            dropoff_longitude=request.form.get('dropoff_longitude'),
            dropoff_latitude=request.form.get('dropoff_latitude'),
            passenger_count=request.form.get('passenger_count')
        )

        pred_df = data.convert_ip_to_df()
        predict_pipeline = PredictPipeline(pred_df)
        result = predict_pipeline.predict()
        return render_template('index.html', result=result)

if __name__=="__main__":
    app.run(host="0.0.0.0")        