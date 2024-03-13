from flask import Flask,request
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import math
import joblib

app=Flask(__name__)

class Battery:
    def __init__(self,voltage,current,temperature,age):
        self.voltage=voltage
        self.current=current
        self.temperature=temperature
        self.age=age

class RandomForestModel:
    def __init__(self):
        self.model=RandomForestRegressor()
    
    def loadModel(self,modelPath):
        self.model=joblib.load(modelPath)

    def predict(self,batteryData):
        X=np.array([batteryData.voltage, batteryData.current, batteryData.temperature, batteryData.age]).reshape(1,-1)
        return self.model.predict(X)

class HealthPrediction:
    def __init__(self,battery,model):
        self.battery=battery
        self.model=model
        result=self.model.predict(self.battery)
        self.healthState=math.ceil(result[0])

@app.route('/predict',methods=['POST'])
def getPrediction():
    data=request.get_json()
    battery=Battery(data['voltage'], data['current'],data['temperature'],data['age'])

    model=RandomForestModel()
    model.loadModel('model.pkl')
    prediction=HealthPrediction(battery,model)

    return {'healthState': prediction.healthState}

if __name__=="__main__":
    app.run(debug=True)
