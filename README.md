# Customer Churn Analysis Agent

An AI-powered customer churn analysis application that combines a trained machine learning model, reusable Python analysis tools, and an LLM agent to answer natural-language questions about customer churn.

The main goal of this project was not just to create a chatbot that talks about a dataset. I wanted specific numerical answers to come from actual computations on the data or from the trained churn model. The LLM is mainly responsible for understanding the user's question, determining what needs to be calculated, and explaining the verified results.

**Live Application:** https://customer-churn-agent-kq8ky8jam2s5sc7cfncyyc.streamlit.app/

---

## Overview

The project uses the Telco Customer Churn dataset containing **7,043 customers and 21 columns**.

I first cleaned and explored the dataset in a Google Colab notebook, then trained and evaluated churn prediction models. The selected model was exported and integrated into a local Python application.

From there, I created reusable data and model tools and connected them to an LLM agent using Groq. The final application is a Streamlit chat interface where a user can ask questions such as:

- Which contract type has the highest churn rate?
- How many customers churned?
- What is the average monthly charge for customers who churned?
- What is the predicted churn risk for a specific customer?
- Which customers currently have the highest predicted churn risk?
- What would a customer's churn risk be if their contract changed?
- Which contract type has the highest churn rate, and what is the average monthly charge for churned customers?

The application supports both single-step and multi-step questions.

---

## Main Features

### Exploratory Data Analysis

The project notebook includes:

- dataset inspection and data type review
- missing and invalid value checks
- duplicate row and customer ID checks
- churn distribution analysis
- numeric feature summaries
- churn rates across categorical features
- tenure and monthly-charge group analysis
- visualizations of important churn relationships

### Churn Prediction

The application uses a trained Logistic Regression pipeline that returns:

- a churn/stay prediction
- a churn probability
- a Python-computed churn percentage

### Reusable Analysis Tools

The agent can perform real computations through Python tools for:

- dataset summaries
- category counts
- churn rates by category
- numeric summaries
- customer lookup
- individual churn prediction
- high-risk customer ranking
- controlled what-if churn prediction

### LLM Agent

The LLM is used to:

- interpret natural-language questions
- create a structured computation plan
- select one or multiple approved tools
- provide the required arguments
- explain verified tool results

### Verification

Tool results are checked before they are used to form the final response.

If a tool fails, returns an error, or produces an empty result, the agent can make one controlled replanning attempt. If the computation still cannot be completed reliably, it stops rather than inventing an answer.

### Streamlit Interface

The final UI includes:

- chat-style questions and responses
- visible conversation history
- expandable analysis details
- tool and argument transparency
- verification status
- a clear-conversation button
- graceful application-level error handling

### Docker Support

The application can run inside a Docker container with its Python environment and dependencies isolated from the host machine.

---

## Project Architecture

I intentionally separated the project into layers rather than putting the UI, LLM logic, dataframe calculations, and machine learning model into one large file.

```text
User
  |
  v
Streamlit UI (app.py)
  |
  v
Agent (src/agent.py)
  |
  +----------------------+
  |                      |
  v                      v
Data Tools           Model Tools
(data_tools.py)      (model_tools.py)
  |                      |
  v                      v
Clean Dataset        Trained Model
```

A normal agent request follows approximately this workflow:

```text
User Question
     |
     v
LLM creates structured plan
     |
     v
Python validates requested tools
     |
     v
Approved Python tools execute
     |
     v
Results are verified
     |
     v
LLM explains verified results
     |
     v
Final Answer
```

This separation was important because I did not want the LLM directly calculating statistics or having unrestricted access to Python execution.

---

## Dataset Cleaning and Data Quality

One of the first things I checked was whether the original dataset had missing values.

A basic Pandas missing-value check initially suggested that the dataset did not contain standard null values. However, I noticed that `TotalCharges` was stored as a text/object column even though it should represent a numeric value.

I tested whether all of its values could actually be converted to numbers and found **11 rows containing blank `TotalCharges` values**.

Those rows belonged to customers with:

```text
tenure = 0
```

which indicated that they were new customers who had not accumulated historical charges yet.

Instead of deleting those customer records, I converted `TotalCharges` to numeric and treated those blank accumulated charges as `0`.

This allowed me to preserve all **7,043 customer records**.

I also checked:

- standard missing values
- duplicate rows
- duplicate customer IDs
- category values
- numeric feature ranges

No duplicate rows or duplicate customer IDs were found, and I did not identify another major data-quality issue that required changing the dataset.

The cleaned dataset still contains:

```text
7,043 rows
21 columns
```

The model uses **19 features**, excluding `customerID` and the prediction target `Churn`.

The complete cleaning and EDA process is documented in:

```text
notebooks/Customer_Churn_Analysis.ipynb
```

The cleaned dataset used by the application is stored in:

```text
data/customer_churn_clean.csv
```

---

## Exploratory Data Analysis Findings

Some of the more noticeable relationships I found were:

- About **26.5%** of customers churned and **73.5%** stayed.
- Customers with month-to-month contracts had much higher churn than customers with longer contracts.
- Month-to-month churn was approximately **42.7%**.
- One-year contract churn was approximately **11.3%**.
- Two-year contract churn was approximately **2.8%**.
- Customers who churned generally had shorter tenure.
- Customers who churned had higher average monthly charges.
- Fiber optic customers showed higher churn than DSL customers.
- Electronic-check customers showed relatively high churn.
- Customers without services such as online security or technical support also showed higher churn.
- Gender and phone service showed relatively small differences in churn.

These are patterns and associations in this dataset and should not automatically be interpreted as causal relationships.

---

## Model Training

I separated the cleaned dataset into:

```text
80% training data
20% testing data
```

using a stratified split.

I used stratification so that the original churn balance was preserved in both sets. The training and testing sets both remained approximately:

```text
73.46% stayed
26.54% churned
```

The final model uses:

```text
4 numeric features
15 categorical features
19 total features
```

The numeric features are:

```text
SeniorCitizen
tenure
MonthlyCharges
TotalCharges
```

Numerical features are standardized using `StandardScaler`.

Categorical features are encoded using `OneHotEncoder`.

These preprocessing operations are combined with the classifier inside a Scikit-learn pipeline, which means future customer predictions receive the same preprocessing that was used during model training.

---

## Model Comparison

I started with **Logistic Regression** because churn is a binary classification problem and Logistic Regression provides probabilities that are useful for customer risk ranking.

I also trained a **Random Forest** to compare the simpler linear model against a more flexible tree-based model.

### Logistic Regression

| Metric | Score |
|---|---:|
| Accuracy | 80.6% |
| Precision | 65.7% |
| Recall | 55.9% |
| F1 Score | 0.604 |
| ROC-AUC | 0.842 |

### Random Forest

| Metric | Score |
|---|---:|
| Accuracy | 78.4% |
| Recall | 48.1% |
| F1 Score | 0.541 |
| ROC-AUC | 0.821 |

For another baseline comparison, simply predicting that every customer will stay would achieve approximately **73.5% accuracy** because the dataset is somewhat imbalanced toward non-churn customers.

The Logistic Regression model performed better than Random Forest across the main metrics I was evaluating, so I selected it as the final model.

I did not treat this as an exhaustive search for the highest possible model score. My goal was to compare a reasonable interpretable baseline against a more flexible model, select the stronger result, and then spend the remaining project time building and testing the agent around it.

---

## Why I Chose ROC-AUC

I used **ROC-AUC as the main overall evaluation metric**.

Accuracy alone can be misleading for this dataset because approximately 73.5% of customers did not churn. A model that predicted "stay" for everyone would therefore appear reasonably accurate without identifying any churn risk.

I monitored several metrics:

**Accuracy** measures the percentage of overall predictions that were correct.

**Precision** measures how often customers predicted to churn actually churned.

**Recall** measures how many customers who actually churned were successfully identified. This matters in a retention use case because a false negative represents a customer who churns without being identified as at risk.

**F1 Score** balances precision and recall.

**ROC-AUC** measures how effectively the model ranks higher-risk customers above lower-risk customers across different classification thresholds.

I chose ROC-AUC as the main overall metric because the final application makes use of predicted probabilities and churn-risk ranking rather than relying only on a fixed yes/no classification.

I still monitored recall because missing actual churners is an important limitation of a retention model.

---

## Reusable Data Tools

After completing the notebook and selecting the model, I moved useful calculations into reusable Python functions.

I did not want the LLM to receive the whole dataframe and try to calculate answers itself.

The current data tools include:

### `get_dataset_summary()`

Returns basic information such as:

- number of customers
- number of columns
- customers who churned
- customers who stayed

### `get_churn_rate_by_category(column)`

Calculates churn percentages for each category in an approved categorical feature.

For example:

```text
Contract
InternetService
PaymentMethod
TechSupport
```

### `get_numeric_summary(column, churn_status=None)`

Calculates:

- count
- average
- median
- minimum
- maximum

for supported numeric columns.

It can optionally filter the calculation to customers who churned or stayed.

### `get_category_counts(column)`

Returns how many customers belong to each category.

### `get_customer_by_id(customer_id)`

Retrieves a real customer record from the cleaned dataset.

---

## Why I Chose These Tools

I considered whether the application should have many more data-analysis functions.

Instead of trying to create a tool for every possible Pandas operation, I focused on the main question types that became useful during EDA and while testing the agent:

```text
dataset summary questions
category comparisons
churn-rate comparisons
numeric summaries
individual customer questions
model predictions
risk ranking
```

This gave the agent useful flexibility without creating an unnecessarily large set of tools for it to choose between.

The architecture is still extensible. New analytical operations can be added later by implementing a trusted Python function and explicitly registering it with the agent.

---

## Model Tools

The trained model is saved at:

```text
models/churn_model.joblib
```

### `predict_churn(customer_data)`

This function receives the model's 19 expected customer features and returns:

- churn/stay prediction
- raw churn probability
- churn probability percentage

### `get_high_risk_customers(customer_df, top_n)`

This function scores multiple customers with the trained model and ranks them by predicted churn probability.

The probabilities come from the trained Scikit-learn pipeline rather than from the LLM.

---

## Agent-Friendly Wrapper Functions

Some Python functions were useful internally but were not ideal interfaces for an LLM.

For example, the core churn prediction function expects all **19 customer features**.

I did not want the LLM to reconstruct 19 values every time someone asked:

> What is the churn risk for customer 7590-VHVEG?

Instead, I created an agent-facing wrapper:

```text
predict_customer_churn(customer_id)
```

Python then:

1. retrieves the real customer from the dataset
2. removes `customerID` and historical `Churn`
3. prepares the expected model features
4. calls the trained model
5. returns the prediction and probability

I used the same idea for high-risk customer ranking.

The underlying ranking function works with a dataframe, but the agent-facing wrapper keeps that dataframe inside Python and only requires the LLM to provide a simple `top_n` value.

This keeps the interface simpler for the LLM while keeping model and dataframe operations controlled.

---

## LLM Agent Design

The application uses **Groq** for LLM inference.

The LLM has two main responsibilities.

### 1. Planning

The first call interprets the user's full question and decides which computation or computations are necessary.

### 2. Explanation

After Python executes and verifies those computations, the LLM receives the results and turns them into a concise natural-language response.

The important distinction is:

```text
LLM decides what needs to be calculated.

Python actually calculates it.
```

A normal successful request therefore usually requires approximately two LLM calls rather than repeatedly calling the model in an open-ended loop.

This also helps keep the agent reasonably efficient under API rate limits.

---

## Multi-Step Questions and Structured Planning

The biggest challenge I ran into during this project was getting the agent to reliably answer questions requiring more than one computation.

For example:

> Which contract type has the highest churn rate, and what is the average monthly charge for customers who churned?

The correct response requires two separate calculations:

```text
get_churn_rate_by_category("Contract")

get_numeric_summary(
    column="MonthlyCharges",
    churn_status="Yes"
)
```

My first implementation used Groq's native tool-calling behavior.

That worked well for simple questions, but during testing I found that a compound question could result in only the first required tool being selected.

For the example above, the agent correctly found the contract churn rates but initially did not calculate the average monthly charge.

The final-answer layer correctly avoided inventing the missing number, but the planning step itself was incomplete.

I changed the planning layer to return a **structured JSON object** describing every required tool call instead.

The planner now produces a structure similar to:

```json
{
  "tool_calls": [
    {
      "tool_name": "get_churn_rate_by_category",
      "arguments": "{\"column\":\"Contract\"}"
    },
    {
      "tool_name": "get_numeric_summary",
      "arguments": "{\"column\":\"MonthlyCharges\",\"churn_status\":\"Yes\"}"
    }
  ]
}
```

Python parses this plan, validates the requested tool names, executes each operation, and collects their results.

This made multi-step questions much more reliable.

---

## Tool Registry and Controlled Execution

The LLM cannot execute arbitrary Python functions.

Approved tool names are mapped to trusted Python functions through a controlled registry.

Conceptually:

```python
TOOL_REGISTRY = {
    "get_dataset_summary": ...,
    "get_category_counts": ...,
    "get_churn_rate_by_category": ...,
    "get_numeric_summary": ...,
    "get_customer_by_id": ...,
    "predict_customer_churn": ...,
    "predict_customer_what_if": ...,
    "get_high_risk_customers": ...
}
```

The LLM can request an operation by name, but Python checks that the requested operation exists in this approved registry before executing it.

This keeps the agent flexible without giving it unrestricted execution access.

---

## Tool Argument Handling

Another issue appeared after moving to structured planning.

For a what-if question, the Python function expected:

```json
{
  "customer_id": "7590-VHVEG",
  "feature": "Contract",
  "new_value": "Two year"
}
```

but an early planner response produced:

```json
{
  "customer_id": "7590-VHVEG",
  "what_if": {
    "Contract": "Two year"
  }
}
```

The tool correctly failed because the function did not accept an argument named `what_if`.

Rather than changing the Python function to accept whatever argument structure the LLM generated, I added a planner guide describing the exact expected argument format for each approved tool.

The LLM still decides which tool is appropriate, but the interface between planning and execution is more predictable.

---

## Verification and Safe Failure Handling

Before computed results are used to generate the final answer, they pass through a verification step.

The verifier checks for situations such as:

- missing results
- explicit tool errors
- empty dictionaries
- empty lists

For example, I tested:

> What is the predicted churn risk for customer NOT-A-REAL-CUSTOMER?

The Python customer lookup returned an error.

Instead of fabricating a churn probability, the application returned a safe failure response.

The workflow allows **one controlled retry** after a verification failure.

I intentionally limited the retry rather than creating an unlimited agent loop. This avoids repeatedly calling the LLM when the underlying request cannot be satisfied and keeps the workflow more predictable.

---

## Numerical Grounding

Another issue I specifically tested was whether the final LLM might accidentally perform its own arithmetic.

During an early what-if test, Python produced:

```text
0.2898
```

The LLM incorrectly converted that into:

```text
29.98%
```

instead of:

```text
28.98%
```

I fixed this by moving the percentage calculation into Python.

The model tool now returns both:

```text
raw probability = 0.2898
percentage = 28.98
```

The final-response prompt is also instructed to use numerical values exactly as they were computed and not create new numerical values through its own arithmetic.

The design is therefore:

```text
Python owns the numbers.
LLM owns the explanation.
```

---

## What-If Churn Predictions

The application supports controlled hypothetical questions for existing customers.

For example:

> What would the predicted churn risk be for customer 7590-VHVEG if their Contract changed to Two year?

This does not generate a random customer or modify the original dataset.

Instead, Python:

1. retrieves the customer's real record
2. saves the original feature value
3. validates that the requested feature can be changed
4. validates the new value
5. creates a temporary copy of the customer record
6. changes only the requested feature
7. sends the modified model features through the trained pipeline
8. returns the new model prediction

The source dataset remains unchanged.

This means the resulting probability is **a model prediction under a hypothetical condition**, not a historical fact from the dataset.

What-if analysis is intentionally limited to a controlled set of categorical features such as:

- `Contract`
- `InternetService`
- `PaperlessBilling`
- `PaymentMethod`

This prevents the agent from sending unsupported values into the model.

For customer `7590-VHVEG`, changing the contract from Month-to-month to Two year produced a model-estimated churn probability of **28.98%** and a predicted class of **Stay**.

---

## Streamlit Application

The final application uses Streamlit as its user interface.

`app.py` is intentionally kept separate from the agent and model implementation.

It mainly handles:

- displaying the page
- receiving user questions
- displaying answers
- storing visible chat history
- showing optional analysis details
- clearing the conversation
- handling unexpected application errors

The UI only needs to call:

```python
answer_question(user_question)
```

It does not need to know the details of Groq planning, dataframe operations, model inference, verification, or retries.

---

## Conversation History

Streamlit reruns the application script after interactions, so I use:

```python
st.session_state
```

to preserve the visible conversation.

This allows the UI to display:

```text
Question 1
Answer 1

Question 2
Answer 2
```

without losing earlier messages after each Streamlit rerun.

This is currently **UI conversation history**, not full multi-turn agent memory.

A follow-up question that depends on an earlier message may still require the user to restate enough context for the planner to understand it.

---

## Analysis Details

Each assistant response contains an optional expandable section that shows:

- which tool or tools were used
- which arguments were passed
- verification status

For a simple question, this might show:

```text
Tool: get_churn_rate_by_category

Arguments:
{
    "column": "Contract"
}

Verification:
Tool results verified successfully.
```

For a compound question, multiple tools appear.

I included this because it makes it easier to inspect where an answer came from without cluttering the normal chat response.

---

## Error Handling

There are two main error-handling layers.

### Agent-Level Errors

Expected computation problems, such as an invalid customer ID or failed tool result, are handled through verification and the bounded retry workflow.

### Application-Level Errors

Unexpected failures, such as an API or application exception, are caught in the Streamlit interface.

Instead of exposing a raw traceback to the user, the application displays a short error message and keeps the UI usable.

---

## Project Structure

```text
customer-churn-agent/
|
|-- app.py
|-- README.md
|-- requirements.txt
|-- Dockerfile
|-- .dockerignore
|-- .gitignore
|
|-- data/
|   |-- customer_churn_clean.csv
|
|-- models/
|   |-- churn_model.joblib
|
|-- notebooks/
|   |-- Customer_Churn_Analysis.ipynb
|
|-- src/
|   |-- __init__.py
|   |-- agent.py
|   |-- data_tools.py
|   |-- model_tools.py
|
|-- test_agent.py
|-- test_customer_workflow.py
|-- test_llm.py
|-- test_model.py
|-- test_risk_ranking.py
```

### Important Files

**`app.py`**  
Streamlit chat interface.

**`src/agent.py`**  
LLM planning, agent-facing wrappers, controlled tool execution, verification, retry logic, and final-response generation.

**`src/data_tools.py`**  
Trusted Pandas computations against the cleaned dataset.

**`src/model_tools.py`**  
Loads the trained model and performs churn prediction and risk ranking.

**`notebooks/Customer_Churn_Analysis.ipynb`**  
Data cleaning, EDA, model training, evaluation, and model selection.

**`models/churn_model.joblib`**  
Saved preprocessing and Logistic Regression pipeline.

**`data/customer_churn_clean.csv`**  
Cleaned dataset used by the application.

---

## Running Locally

### 1. Clone the repository

```bash
git clone <repository-url>
cd customer-churn-agent
```

### 2. Create a virtual environment

Python **3.12** is recommended.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure Groq

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_groq_api_key
```

The `.env` file is excluded from Git.

### 5. Start the application

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

## Running with Docker

Build the image:

```bash
docker build -t customer-churn-agent .
```

Run the container while providing the Groq key at runtime:

```bash
docker run --env-file .env -p 8501:8501 customer-churn-agent
```

Then open:

```text
http://localhost:8501
```

The `.env` file is excluded through `.dockerignore`, so the API key is not copied into the Docker image.

I tested the Dockerized application end-to-end and confirmed that it could:

- start Streamlit
- load the cleaned dataset
- load the saved model
- connect to Groq
- plan and execute a tool
- verify the tool result
- display the final answer through the browser

---

## Example Questions

```text
How many customers are in the dataset?
```

```text
How many customers churned?
```

```text
Which contract type has the highest churn rate?
```

```text
What is the average monthly charge for customers who churned?
```

```text
How many customers use each internet service type?
```

```text
What is the predicted churn risk for customer 7590-VHVEG?
```

```text
Who are the five customers with the highest predicted churn risk?
```

```text
Which contract type has the highest churn rate, and what is the average monthly charge for customers who churned?
```

```text
What would the predicted churn risk be for customer 7590-VHVEG if their Contract changed to Two year?
```

---

## Testing

I tested the project incrementally rather than waiting until the end.

Tests included:

- loading the trained model outside the notebook
- predicting a real customer's churn risk
- comparing a prediction with the customer's historical outcome
- dataset summary calculations
- churn rates by category
- numerical summaries
- customer lookup
- high-risk customer ranking
- simple LLM tool planning
- multi-tool questions
- invalid customer IDs
- verification failure behavior
- what-if prediction
- numerical grounding
- Streamlit integration
- Dockerized end-to-end execution

Several of these tests directly led to changes in the final architecture, especially the multi-tool structured planner and the decision to move percentage calculations entirely into Python.

---

## Limitations and What I Would Improve

### Model Performance

The final Logistic Regression model has a recall of approximately **55.9%**, so it still misses some customers who eventually churn.

With more time, I would experiment further with the model and its decision threshold to see if I could identify more true churners without creating too many false warnings. I would also compare a few additional model types rather than limiting the comparison to Logistic Regression and Random Forest.

### More Flexible Analysis

The current agent uses a controlled set of analysis tools covering the main questions I wanted the application to answer.

I would make these tools more flexible so users could combine multiple filters and ask more detailed segment questions without requiring a separate Python function for every type of analysis.

I would also consider generating charts for questions where a visual comparison would communicate the result more clearly than text.

### Multi-Turn Context

The Streamlit interface preserves visible conversation history, but the agent currently treats each new question mostly independently.

A future version could provide relevant previous conversation context to the planner so follow-up questions could be understood without repeating the original subject.

### Broader Hypothetical Input

The current what-if tool changes one approved feature for an existing real customer at a time.

I would expand this to support multiple controlled feature changes and eventually allow a user to provide a completely new hypothetical customer profile for evaluation.

---

## AI Tool Usage

I used ChatGPT throughout development as a learning and development assistant.

I mainly used it to:

- understand unfamiliar agent and LLM concepts
- discuss architecture and implementation decisions
- debug Python, environment, Docker, and Groq-related issues
- reason through tool-calling and structured planning behavior
- review test results and edge cases
- improve code comments and project documentation

I tried to understand the code and the reasoning behind each major decision rather than treating generated code as automatically correct. I tested the components locally and changed the implementation when the results exposed problems.

For example, testing showed that the original agent-planning approach could miss part of a multi-step question, which led me to replace that workflow with structured JSON planning. I also moved percentage calculations into Python after observing an incorrect numerical conversion in an LLM-generated response.

---

## Final Design

The main principle behind the final application is:

```text
LLM plans
     |
Python executes
     |
Python verifies
     |
LLM explains
```

The LLM provides a natural-language interface over a controlled set of real computations rather than replacing the analysis layer itself.

This keeps specific numerical claims grounded in the dataset or trained model while still allowing users to interact with the analysis naturally.