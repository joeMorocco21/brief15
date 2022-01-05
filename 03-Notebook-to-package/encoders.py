import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from data import cleanData
from utils import haversine_vectorized
df = cleanData()
class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
    """
        Extracts the day of week (dow), the hour, the month and the year from a time column.
        Returns a copy of the DataFrame X with only four columns: 'dow', 'hour', 'month', 'year'.
    """

    def __init__(self, time_column, time_zone_name='America/New_York'):
       # A COMPLETER
                self.time_column = time_column
                self.time_zone_name = time_zone_name
    def extract_time_features(self, X):
        timezone_name = self.time_zone_name
        time_column = self.time_column
        df = X.copy()
        df.index = pd.to_datetime(df[time_column])
        df.index = df.index.tz_convert(timezone_name)
        df["dow"] = df.index.weekday
        df["hour"] = df.index.hour
        df["month"] = df.index.month
        df["year"] = df.index.year        
        return df
    def fit(self, X, y=None):
        # A COMPLETER
        return self
    def transform(self, X, y=None):
        # A COMPLETER 
        return self.extract_time_features(X)[['dow', 'hour', 'month', 'year']].reset_index(drop=True)


class DistanceTransformer(BaseEstimator, TransformerMixin):
    """
        Computes the haversine distance between two GPS points.
        Returns a copy of the DataFrame X with only one column: 'distance'.
    """

    def __init__(self,
                 start_lat="pickup_latitude",
                 start_lon="pickup_longitude",
                 end_lat="dropoff_latitude",
                 end_lon="dropoff_longitude"):
        # A COMPPLETER
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
    
        return pd.DataFrame(haversine_vectorized(X)).rename(columns={0: "course distance [km]"}).copy()

