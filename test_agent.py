from src.agent import answer_question

question = (
    "What would the predicted churn risk be for customer 7590-VHVEG "
    "if their Contract changed to Two year?"
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