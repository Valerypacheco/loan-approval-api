# train.py = Model training pipeline

# Import required libraries for data processing, modeling, and evaluation

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Load dataset from CSV file

df = pd.read_csv("data/loan_data.csv")
# print(df.columns)
# Ver los valores únicos en la columna 'Employment Status'
# for column in df.columns:
#     print(f'valores unicos para {column}')
#     print(df[column].unique())

# Delete Text column
df = df.drop(columns=['Text'])

# Convert the approval column into binary
df['Approval'] = df['Approval'].map({'Approved': 1, 'Rejected': 0})

# Convert the Employment_Status column into binary
df['Employment_Status'] = df['Employment_Status'].map({'employed': 1, 'unemployed': 0})


# Separate input features (X) and target variable (y)

X = df.drop("Approval", axis=1)
y = df["Approval"]

# Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create a machine learning pipeline

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    ))
])

# Train the model using training data

pipeline.fit(X_train, y_train)

# Generate predictions and probabilities

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

# Evaluate model performance
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

# Save trained model to file
joblib.dump(pipeline, "model/loan_pipeline.pkl")