from data import cleanData
from utils import haversine_vectorized, compute_rmse
from encoders import TimeFeaturesEncoder, DistanceTransformer
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
class Trainer():
    def __init__(self, X, y, **kwargs):
        self.X_train = X
        self.y_train = y
    def set_pipeline(self):
        dist_pipe = Pipeline([('distance', DistanceTransformer()),('distance scale',StandardScaler() )])
        pipeTime = Pipeline([('time_features_create', TimeFeaturesEncoder('pickup_datetime')),('time_features_ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))])
        preprocessor = ColumnTransformer([
            ('distance',dist_pipe, ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']),
            ('time', pipeTime, ['pickup_datetime'])])
        pipe = Pipeline([
            ('dist_and_time', preprocessor),
            ('regLinear', LinearRegression())
        ])
        return pipe

    def run(self,X_train, y_train, pipe):
        pipeline = pipe.fit(X_train, y_train)
        return pipeline

    def evaluate(self,X_test, y_test, pipeline):
        y_pred = pipeline.predict(X_test)
        rmse = np.sqrt(np.mean(np.square(y_test - y_pred)))
        print(rmse)
        return rmse


df = cleanData()
X = df.drop(["fare_amount"], axis=1)
y = df["fare_amount"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
trainer = Trainer(X_train,y_train)
pip = trainer.set_pipeline()
pip = trainer.run(X_train,y_train, pip)
trainer.evaluate(X_test, y_test, pip)