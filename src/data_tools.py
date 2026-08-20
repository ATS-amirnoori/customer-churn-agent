# Load the cleaned customer dataset so reusable analysis tools can query real data

import pandas as pd

DATA_PATH = "data/customer_churn_clean.csv"

df = pd.read_csv(DATA_PATH)


def get_dataset_summary():
    """Return basic information about the customer dataset."""

    return {
        "customers": len(df),
        "columns": len(df.columns),
        "churned_customers": int((df["Churn"] == "Yes").sum()),
        "stayed_customers": int((df["Churn"] == "No").sum())
    }

def get_churn_rate_by_category(column):
    """Return churn rate percentages for each category in a supported column."""

    supported_columns = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod"
    ]

    if column not in supported_columns:
        return {
            "error": f"Unsupported category column: {column}"
        }

    # Convert Churn Yes/No into a temporary 1/0 flag, group customers by the requested category,
    # then average the flag within each group. Since churn = 1 and stay = 0, the average equals
    # the proportion that churned; multiplying by 100 converts that proportion into a percentage.
    churn_rates = (
        df.assign(churn_flag=(df["Churn"] == "Yes").astype(int))
        .groupby(column)["churn_flag"]
        .mean()
        .mul(100)
        .round(2)
    )

    return churn_rates.to_dict()

# Summarize a numeric customer feature using real dataframe calculations.
# The optional churn_status filter lets the same tool analyze all customers,
# only customers who churned, or only customers who stayed.
def get_numeric_summary(column, churn_status=None):
    """Return summary statistics for a supported numeric column, optionally filtered by churn status."""

    supported_columns = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    if column not in supported_columns:
        return {
            "error": f"Unsupported numeric column: {column}"
        }

    filtered_df = df

    if churn_status is not None:
        if churn_status not in ["Yes", "No"]:
            return {
                "error": "churn_status must be 'Yes', 'No', or None"
            }

        filtered_df = df[df["Churn"] == churn_status]

    values = filtered_df[column]

    return {
        "count": int(values.count()),
        "average": round(float(values.mean()), 2),
        "median": round(float(values.median()), 2),
        "minimum": round(float(values.min()), 2),
        "maximum": round(float(values.max()), 2)
    }

# Count how many customers belong to each category in a supported column.
# This lets the agent answer questions about category sizes using the real dataset.

def get_category_counts(column):
    """Return the number of customers in each category of a supported column."""

    supported_columns = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "Churn"
    ]

    if column not in supported_columns:
        return {
            "error": f"Unsupported category column: {column}"
        }

    counts = df[column].value_counts()

    return {
        str(category): int(count)
        for category, count in counts.items()
    }

# Look up one customer by ID so other tools can use that customer's real dataset values.
# Find the row whose customerID matches the requested ID. If no customer is found,
# return a clear error; otherwise convert the customer's full row into a dictionary
# so the record can easily be passed to other application tools.

def get_customer_by_id(customer_id):
    """Return one customer's record using their customer ID."""

    customer = df[df["customerID"] == customer_id]

    if customer.empty:
        return {
            "error": f"Customer not found: {customer_id}"
        }

    return customer.iloc[0].to_dict()

if __name__ == "__main__":
    print("Dataset summary:")
    print(get_dataset_summary())

    print("\nChurn rate by contract:")
    print(get_churn_rate_by_category("Contract"))

    print("\nMonthly charges for churned customers:")
    print(get_numeric_summary("MonthlyCharges", "Yes"))

    print("\nCustomer counts by internet service:")
    print(get_category_counts("InternetService"))

    print("\nCustomer lookup:")
    print(get_customer_by_id("7590-VHVEG"))