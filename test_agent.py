# Test the completed agent workflow across different question types,
# including a question that should require multiple computation tools.

from src.agent import answer_question


test_questions = [
    (
        "Which contract type has the highest churn rate, "
        "and what is the average monthly charge for customers who churned?"
    )
]


for question in test_questions:

    print("\n" + "=" * 70)
    print("QUESTION:")
    print(question)

    result = answer_question(question)

    print("\nANSWER:")
    print(result["answer"])

    print("\nTOOLS USED:")
    for tool_result in result["tool_results"]:
        print(
            "-",
            tool_result["tool"],
            tool_result["arguments"]
        )

    print("\nVERIFICATION:")
    print(result["verification"])