import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ( RandomForestClassifier, RandomForestRegressor)
from sklearn.cluster import KMeans
from sklearn.metrics import ( accuracy_score, r2_score, silhouette_score )


def load_dataset(file_path):
  df = pd.read_csv(file_path)
  return df


def detect_task(df, target_column=None):
    if target_column is None:
        return "clustering"
    unique_values = df[target_column].nunique()
    if unique_values < 20 and df[target_column].dtype == 'object':
        return "classification"
    elif np.issubdtype(df[target_column].dtype, np.number):
        return "regression"
    return "unknown"


def preprocess_data(df, target_column=None):
    df = df.copy()
    for col in df.select_dtypes(include=['object']).columns:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))
    if target_column:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        imputer = SimpleImputer(strategy='mean')
        scaler = StandardScaler()
        X = imputer.fit_transform(X)
        X = scaler.fit_transform(X)
        return X, y
    else:
        imputer = SimpleImputer(strategy='mean')
        scaler = StandardScaler()
        X = imputer.fit_transform(df)
        X = scaler.fit_transform(X)
        return X


def recommend_model(task_type):
    if task_type == "classification":
        return RandomForestClassifier()
    elif task_type == "regression":
        return RandomForestRegressor()
    elif task_type == "clustering":
        return KMeans(n_clusters=3)


def train_and_evaluate(df, target_column=None):
    task = detect_task(df, target_column)
    print("Detected Task:", task)
    if task == "classification":
        X, y = preprocess_data(df, target_column)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = recommend_model(task)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, predictions))
    elif task == "regression":
        X, y = preprocess_data(df, target_column)
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
        model = recommend_model(task)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print("R2 Score:", r2_score(y_test, predictions))
    elif task == "clustering":
        X = preprocess_data(df)
        model = recommend_model(task)
        clusters = model.fit_predict(X)
        print("Silhouette Score:", silhouette_score(X, clusters))