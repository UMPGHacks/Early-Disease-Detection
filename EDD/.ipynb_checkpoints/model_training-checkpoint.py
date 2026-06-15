from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR.parent / "dataset"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_symptom_model():
    data = pd.read_csv(DATASET_DIR / "improved_disease_dataset.csv")
    X = data.drop(columns=["disease"])
    y = data["disease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    joblib.dump(model, MODELS_DIR / "symptom_model.pkl")
    return accuracy


def train_heart_model():
    data = pd.read_csv(DATASET_DIR / "heart_failure_clinical_records_dataset.csv")
    X = data.drop(columns=["DEATH_EVENT"])
    y = data["DEATH_EVENT"]

    numeric_features = X.columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=300, random_state=42)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    joblib.dump(model, MODELS_DIR / "heart_model.pkl")
    return accuracy


def train_liver_model():
    data = pd.read_csv(DATASET_DIR / "Indian Liver Patient Dataset (ILPD).csv")
    data["A/G Ratio"] = data["A/G Ratio"].fillna(data["A/G Ratio"].median())

    X = data.drop(columns=["Selector"])
    y = data["Selector"].replace({2: 0, 1: 1})

    numeric_features = [
        "Age",
        "TB",
        "DB",
        "Alkphos",
        "Sgpt",
        "Sgot",
        "TP",
        "ALB",
        "A/G Ratio",
    ]
    categorical_features = ["Gender"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=300, random_state=42)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    joblib.dump(model, MODELS_DIR / "liver_model.pkl")
    return accuracy


def main():
    symptom_accuracy = train_symptom_model()
    heart_accuracy = train_heart_model()
    liver_accuracy = train_liver_model()

    print(f"Symptom model accuracy: {symptom_accuracy:.4f}")
    print(f"Heart model accuracy: {heart_accuracy:.4f}")
    print(f"Liver model accuracy: {liver_accuracy:.4f}")
    print(f"Saved models to: {MODELS_DIR}")


if __name__ == "__main__":
    main()
