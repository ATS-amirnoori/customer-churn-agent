# Test the saved churn model locally using one real customer from the cleaned dataset

import pandas as pd
from src.model_tools import predict_churn

# Load the cleaned customer dataset
df = pd.read_csv("data/customer_churn_clean.csv")

# Use the first customer as a simple test
customer = df.iloc[0]

# Save the real outcome before removing columns the model does not use
actual_outcome = customer["Churn"]

# Prepare only the 19 features expected by the model
customer_features = customer.drop(["customerID", "Churn"]).to_dict()

# Run the reusable prediction tool
result = predict_churn(customer_features)

print("Customer ID:", customer["customerID"])
print("Prediction:", result["prediction"])
print("Churn probability:", f'{result["churn_probability"] * 100:.2f}%')
print("Actual outcome:", actual_outcome)