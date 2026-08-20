# Load the saved churn model and provide a reusable churn prediction tool

import joblib
import pandas as pd

MODEL_PATH = "models/churn_model.joblib"

model = joblib.load(MODEL_PATH)


def predict_churn(customer_data):
    """Predict churn risk for one customer."""

    customer_df = pd.DataFrame([customer_data])

    churn_probability = model.predict_proba(customer_df)[0, 1]
    churn_prediction = model.predict(customer_df)[0]

    probability = float(churn_probability)

    return {
        "prediction": "Churn" if churn_prediction == 1 else "Stay",
        "churn_probability": round(probability, 4),
        "churn_probability_percent": round(probability * 100, 2)
    }

# Score multiple customers with the trained model, sort them by predicted churn probability,
# and return the customers with the highest estimated churn risk.

def get_high_risk_customers(customer_df, top_n=10):
    """Return the customers with the highest predicted churn probabilities."""

    model_features = customer_df.drop(
        columns=["customerID", "Churn"],
        errors="ignore"
    )

    churn_probabilities = model.predict_proba(model_features)[:, 1]

    results = customer_df.copy()
    results["churn_probability"] = churn_probabilities

    results = results.sort_values(
        by="churn_probability",
        ascending=False
    )

    columns_to_return = [
        "customerID",
        "tenure",
        "Contract",
        "InternetService",
        "MonthlyCharges",
        "churn_probability"
    ]

    return results[columns_to_return].head(top_n)

if __name__ == "__main__":
    print("Churn model loaded successfully.")