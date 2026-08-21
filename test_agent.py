from src.agent import answer_question

question = (
    "Which contract type has the highest churn rate, "
    "and what is the average monthly charge for customers who churned?"
)

result = answer_question(question)

print("QUESTION:")
print(question)

print("\nANSWER:")
print(result["answer"])

print("\nTOOLS USED:")
for tool_result in result["tool_results"]:
    print(
        "-",
        tool_result["tool"],
        tool_result["arguments"],
        "->",
        tool_result["result"]
    )

print("\nVERIFICATION:")
print(result["verification"])