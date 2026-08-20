# Define the controlled toolbox that the AI agent is allowed to use.

import pandas as pd

from src.data_tools import (
    get_dataset_summary,
    get_category_counts,
    get_churn_rate_by_category,
    get_numeric_summary,
    get_customer_by_id
)

from src.model_tools import (
    predict_churn,
    get_high_risk_customers
)


DATA_PATH = "data/customer_churn_clean.csv"

customer_df = pd.read_csv(DATA_PATH)


def rank_high_risk_customers(top_n=10):
    """Return the customers with the highest model-predicted churn risk."""

    results = get_high_risk_customers(
        customer_df,
        top_n=top_n
    )

    return results.to_dict(orient="records")

# Look up a customer by ID and pass their real dataset features to the trained
# churn model, giving the agent a simple and reliable prediction tool.

def predict_customer_churn(customer_id):
    """Return model-predicted churn risk for a customer ID."""

    customer = get_customer_by_id(customer_id)

    if "error" in customer:
        return customer

    customer_features = {
        key: value
        for key, value in customer.items()
        if key not in ["customerID", "Churn"]
    }

    prediction = predict_churn(customer_features)

    return {
        "customer_id": customer_id,
        "prediction": prediction["prediction"],
        "churn_probability": prediction["churn_probability"]
    }

# Map simple tool names to the real Python functions they are allowed to execute.
# The LLM may request a tool by name, but Python only executes functions listed
# here, keeping all dataframe and model operations controlled and predictable.
TOOL_REGISTRY = {
    "get_dataset_summary": get_dataset_summary,
    "get_category_counts": get_category_counts,
    "get_churn_rate_by_category": get_churn_rate_by_category,
    "get_numeric_summary": get_numeric_summary,
    "get_customer_by_id": get_customer_by_id,
    "predict_customer_churn": predict_customer_churn,
    "get_high_risk_customers": rank_high_risk_customers
}


if __name__ == "__main__":
    print("Available agent tools:")

    for tool_name in TOOL_REGISTRY:
        print("-", tool_name)