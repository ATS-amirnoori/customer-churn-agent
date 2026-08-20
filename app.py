# Streamlit interface for the customer churn analysis agent.

import streamlit as st

from src.agent import answer_question


st.set_page_config(
    page_title="Customer Churn Analysis Agent",
    page_icon="📊"
)


st.title("Customer Churn Analysis Agent")

st.write(
    "Ask questions about the customer dataset or request churn-risk predictions."
)


# Use a form so the agent runs only when the user intentionally submits a question.
with st.form("question_form"):

    user_question = st.text_input(
        "Ask a question:",
        placeholder="Which contract type has the highest churn rate?"
    )

    submitted = st.form_submit_button("Ask")


if submitted:

    if not user_question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Analyzing your question..."):

            result = answer_question(user_question)

        st.subheader("Answer")

        st.markdown(result["answer"])