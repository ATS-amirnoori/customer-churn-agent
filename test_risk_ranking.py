# Test the high-risk ranking tool using the cleaned customer dataset

import pandas as pd
from src.model_tools import get_high_risk_customers

df = pd.read_csv("data/customer_churn_clean.csv")

high_risk_customers = get_high_risk_customers(df, top_n=5)

print(high_risk_customers.to_string(index=False))