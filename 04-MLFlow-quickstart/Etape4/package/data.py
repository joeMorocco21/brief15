import pandas as pd
def cleanData():
    df = pd.read_csv("C:\\Users\\Joe\\Desktop\\py\\workflow\\train.csv", nrows=1000)
    df = df.drop(["key"], axis=1)
    df = df.dropna(how='any', axis='rows')
    df = df[(df['pickup_latitude']< 90) & (df['pickup_latitude']> -90) & (df['dropoff_latitude']<90) & (df['dropoff_latitude']> -90)]
    df = df.reset_index(drop=True) 
    return df
