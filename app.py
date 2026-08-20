# Basic Streamlit interface for the customer churn analysis agent.

import streamlit as st


st.set_page_config(
    page_title="Customer Churn Analysis Agent",
    page_icon="📊"
)


st.title("Customer Churn Analysis Agent")

st.write(
    "Ask questions about the customer dataset or request churn-risk predictions."
)


user_question = st.text_input(
    "Ask a question:",
    placeholder="Which contract type has the highest churn rate?"
)


if user_question:
    st.write("You asked:")
    st.write(user_question)