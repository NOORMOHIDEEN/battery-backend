from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

df=pd.read_csv('battery_data.csv')

X=df[['voltage','current','temperature','age']]
y=df['healthState']

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor()
rf.fit(X_train, y_train)
joblib.dump(rf,'model.pkl')