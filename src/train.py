import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import json
import time

# --- constants ---
MODEL_DIR = 'models'
DATA_PATH = 'data/dataset.csv'
MODEL_PATH = os.path.join(MODEL_DIR, 'churn_model_v1.pkl')
META_PATH = os.path.join(MODEL_DIR, 'churn_model_v1_meta.json')

# loading data(in reality load data from a csv file or query SQL database)
def load_data():
    print("Loading data...")
    # Creating dummy dataset for churn(0 = not churn, 1 = churn)
    data = {
        'CreditScore': [600, 650, 700, 750, 800, 850, 900, 950] * 20,
        'Age': [25, 30, 35, 40, 45, 50, 55, 60] * 20,
        'Balance': [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000] * 20,
        'Tenure': [1, 2, 3, 4, 5, 6, 7, 8] * 20,
        'Churn': [0, 0, 0, 1, 1, 0, 1, 1] * 20
    }
    df = pd.DataFrame(data)
    return df

# training the model
def train_model():
    df = load_data()
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    #train test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

    #model training
    print("Training model ...")
    model = RandomForestClassifier(n_estimators=100, random_state=7)
    model.fit(X_train, y_train)

    #model evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    #Save the model
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    #Dave metadata
    metadata = {
        "version": "1.0",
        "algorithm" : "RandomForestClassifier",
        "accuracy": accuracy,
        "trained_at": time.time(),
        "features": list(X.columns)
    }
    with open(META_PATH, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to {META_PATH}")

if __name__ == "__main__":
    train_model()