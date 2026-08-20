# Define the controlled toolbox that the AI agent is allowed to use.

import pandas as pd
import os
import json

from dotenv import load_dotenv
from groq import Groq

# Load the private Groq API key and create the LLM client used for agent planning.

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY was not found in the .env file.")

client = Groq(api_key=api_key)

MODEL_NAME = "openai/gpt-oss-20b"

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

# Describe each approved tool to the LLM. These schemas explain what each tool
# does and which arguments are allowed, while TOOL_REGISTRY controls the actual
# Python functions that can execute.

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_dataset_summary",
            "description": (
                "Get basic information about the customer churn dataset, "
                "including total customers and counts of customers who churned or stayed."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "get_category_counts",
            "description": (
                "Count how many customers belong to each category of a supported "
                "categorical column."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "enum": [
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
                        ],
                        "description": "Categorical customer column to count."
                    }
                },
                "required": ["column"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "get_churn_rate_by_category",
            "description": (
                "Calculate the actual churn percentage for every category in a "
                "supported customer column."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "enum": [
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
                        ],
                        "description": "Category to compare churn rates across."
                    }
                },
                "required": ["column"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "get_numeric_summary",
            "description": (
                "Calculate count, average, median, minimum, and maximum for a "
                "numeric customer feature. Can optionally analyze only customers "
                "who churned or only customers who stayed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "enum": [
                            "tenure",
                            "MonthlyCharges",
                            "TotalCharges"
                        ],
                        "description": "Numeric feature to summarize."
                    },
                    "churn_status": {
                        "type": "string",
                        "enum": ["Yes", "No"],
                        "description": (
                            "Optional churn filter. Yes means customers who churned; "
                            "No means customers who stayed."
                        )
                    }
                },
                "required": ["column"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "get_customer_by_id",
            "description": (
                "Retrieve the real dataset record for one customer using their customer ID."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "Customer ID, such as 7590-VHVEG."
                    }
                },
                "required": ["customer_id"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "predict_customer_churn",
            "description": (
                "Use the trained churn model to predict churn risk for a customer "
                "identified by customer ID. Returns a Stay/Churn prediction and probability."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "Customer ID to evaluate."
                    }
                },
                "required": ["customer_id"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "get_high_risk_customers",
            "description": (
                "Rank customers by model-predicted churn probability and return "
                "the customers with the highest estimated churn risk."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "top_n": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "description": "Number of highest-risk customers to return."
                    }
                },
                "required": []
            }
        }
    }
]

AGENT_SYSTEM_PROMPT = """
You are a customer churn analysis agent.

Your job is to understand the user's full question and choose all approved
tools needed to answer it completely.

Important rules:
- Use tools for any question that requires dataset statistics, customer data,
  or churn model predictions.
- Never invent dataset values, percentages, counts, customer information,
  or model probabilities.
- Choose only from the tools provided to you.
- Do not calculate statistics yourself when a tool can calculate them.
- Read the entire question before choosing tools.
- If the user asks multiple questions or requests multiple pieces of
  information, select every tool needed to answer every part.
- Multiple independent tool calls may be returned in the same plan.
- Do not stop after answering only one part of a multi-part question.

Example:
If the user asks for churn rate by contract AND average MonthlyCharges for
customers who churned, request both:
1. get_churn_rate_by_category with column="Contract"
2. get_numeric_summary with column="MonthlyCharges" and churn_status="Yes"
"""

# Require the planner to return a structured list of every computation needed
# to fully answer the user's question.

PLANNER_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "agent_tool_plan",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "tool_calls": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool_name": {
                                "type": "string",
                                "enum": [
                                    "get_dataset_summary",
                                    "get_category_counts",
                                    "get_churn_rate_by_category",
                                    "get_numeric_summary",
                                    "get_customer_by_id",
                                    "predict_customer_churn",
                                    "get_high_risk_customers"
                                ]
                            },
                            "arguments": {
                                "type": "string"
                            }
                        },
                        "required": ["tool_name", "arguments"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["tool_calls"],
            "additionalProperties": False
        }
    }
}

# Ask the LLM to create a structured plan containing every computation needed
# to fully answer the user's question. This step plans only; Python will still
# validate and execute the requested tools separately.

def plan_tool_calls(user_question):
    """Create a structured plan containing every tool needed for the question."""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    AGENT_SYSTEM_PROMPT
                    + "\nReturn every required tool call in tool_calls. "
                    "The arguments field must contain a JSON object encoded as a string. "
                    "For example: "
                    '\'{"column": "Contract"}\'.'
                )
            },
            {
                "role": "user",
                "content": user_question
            }
        ],
        response_format=PLANNER_RESPONSE_FORMAT,
        temperature=0
    )

    return json.loads(response.choices[0].message.content)

# Validate and execute a tool requested by the LLM using only functions in the
# approved registry. The LLM chooses the operation, but Python controls execution.

def execute_tool_call(tool_name, arguments):
    """Execute one approved tool request and return its real result."""

    if tool_name not in TOOL_REGISTRY:
        return {
            "error": f"Tool not allowed: {tool_name}"
        }

    if not isinstance(arguments, dict):
        return {
            "error": f"Invalid arguments for tool: {tool_name}"
        }

    tool_function = TOOL_REGISTRY[tool_name]

    try:
        return tool_function(**arguments)

    except Exception as error:
        return {
            "error": f"Tool execution failed: {str(error)}"
        }

# Check whether the computed tool results are usable before allowing the agent
# to generate a final answer. Errors or empty results trigger a controlled retry.

def verify_tool_results(tool_results):
    """Check that all tool calls returned usable results."""

    if not tool_results:
        return False, "No tool results were produced."

    for item in tool_results:
        result = item["result"]

        if result is None:
            return False, f"{item['tool']} returned no result."

        if isinstance(result, dict) and "error" in result:
            return False, result["error"]

        if isinstance(result, (dict, list)) and len(result) == 0:
            return False, f"{item['tool']} returned an empty result."

    return True, "Tool results verified successfully."

# Convert trusted tool results into a clear natural-language answer.
# The LLM may explain the computed values, but it must not add unsupported numbers.

def generate_final_answer(user_question, tool_results):
    """Generate a grounded response using only the computed tool results."""

    result_text = json.dumps(tool_results, indent=2)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a customer churn analysis assistant. "
                    "Answer the user's question using only the computed tool results provided. "
                    "Do not invent, estimate, or add any numbers that are not present in the results. "
                    "If the results do not contain enough information to answer the question, say so. "
                    "Keep the response clear and concise."
                )
            },
            {
                "role": "user",
                "content": (
                    f"User question:\n{user_question}\n\n"
                    f"Computed tool results:\n{result_text}"
                )
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content

# Run the complete agent workflow: create a structured plan, execute only
# approved tools, verify the results, retry once if needed, and generate
# a grounded final answer.

def answer_question(user_question, max_retries=1):
    """Answer a question using planning, trusted tools, and verification."""

    retry_feedback = None
    tool_results = []

    for attempt in range(max_retries + 1):

        planning_question = user_question

        if retry_feedback:
            planning_question += (
                "\n\nThe previous tool attempt failed verification. "
                f"Reason: {retry_feedback}. "
                "Create a corrected tool plan."
            )

        plan = plan_tool_calls(planning_question)

        planned_calls = plan.get("tool_calls", [])

        if not planned_calls:
            return {
                "answer": "I could not identify a supported computation for this question.",
                "tool_results": [],
                "verification": "No tool calls were planned."
            }

        tool_results = []

        for tool_call in planned_calls:

            tool_name = tool_call["tool_name"]

            try:
                arguments = json.loads(tool_call["arguments"])

            except json.JSONDecodeError:
                arguments = {}
                result = {
                    "error": f"Invalid JSON arguments returned for {tool_name}."
                }

            else:
                result = execute_tool_call(
                    tool_name,
                    arguments
                )

            tool_results.append({
                "tool": tool_name,
                "arguments": arguments,
                "result": result
            })

        verified, verification_message = verify_tool_results(
            tool_results
        )

        if verified:

            final_answer = generate_final_answer(
                user_question,
                tool_results
            )

            return {
                "answer": final_answer,
                "tool_results": tool_results,
                "verification": verification_message
            }

        retry_feedback = verification_message

    return {
        "answer": (
            "I could not produce a reliable answer because the tool results "
            "failed verification."
        ),
        "tool_results": tool_results,
        "verification": retry_feedback
    }

if __name__ == "__main__":

    test_question = (
        "Which contract type has the highest churn rate, "
        "and what is the average monthly charge for customers who churned?"
    )

    result = answer_question(test_question)

    print("Question:")
    print(test_question)

    print("\nAnswer:")
    print(result["answer"])

    print("\nTools used:")
    for tool_result in result["tool_results"]:
        print(
            "-",
            tool_result["tool"],
            tool_result["arguments"]
        )

    print("\nVerification:")
    print(result["verification"])