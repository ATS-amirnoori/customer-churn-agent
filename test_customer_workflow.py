# Test a multi-tool workflow: look up a real customer by ID, then use their
# dataset features to calculate churn risk with the trained model.

from src.data_tools import get_customer_by_id
from src.model_tools import predict_churn

customer_id = "7590-VHVEG"

# Step 1: Retrieve the customer's real record from the dataset
customer = get_customer_by_id(customer_id)

if "error" in customer:
    print(customer["error"])

else:
    # Save the known historical outcome for comparison
    actual_outcome = customer["Churn"]

    # Remove fields the model was not trained to use
    customer_features = {
        key: value
        for key, value in customer.items()
        if key not in ["customerID", "Churn"]
    }

    # Step 2: Send the customer's features to the trained churn model
    prediction = predict_churn(customer_features)

    print("Customer ID:", customer_id)
    print("Predicted outcome:", prediction["prediction"])
    print(
        "Churn probability:",
        f'{prediction["churn_probability"] * 100:.2f}%'
    )
    print("Actual historical outcome:", actual_outcome)