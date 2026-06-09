from sklearn.cluster import KMeans
import pandas as pd

def segment_customers(df):
    X = df[["annual_income", "spending_score"]]

    model = KMeans(n_clusters=3, random_state=42)
    df["segment"] = model.fit_predict(X)

    return df, model