import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer

def get_data():
    # Reading
    df = pd.read_csv("cardiotocography_v2.csv", sep=",")

    # Exploration
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())
    print(df["CLASS"].value_counts())

    # Preparing
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Splitting
    X = df.drop(columns="CLASS")
    y = df["CLASS"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

    # Standardization
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    # Normalization
    normalizer = Normalizer()
    X_train_norm = normalizer.fit_transform(X_train)
    X_test_norm = normalizer.transform(X_test)

    # Result - explored data and 3 sets of X data to compare classification between
    return {
        "original": (X_train, X_test),
        "standardized": (X_train_std, X_test_std),
        "normalized": (X_train_norm, X_test_norm),
        "labels": (y_train, y_test)
    }
