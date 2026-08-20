from src.agent import answer_question

question = "What is the predicted churn risk for customer NOT-A-REAL-CUSTOMER?"

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