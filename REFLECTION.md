# Reflection

The hardest part of this project for me was building the agent workflow around the model. The data cleaning, EDA, and machine learning portions were more familiar to me, but Groq, structured planning, tool execution, and verification were much newer.

The biggest challenge was getting the agent to reliably handle questions requiring multiple calculations. My first approach worked for simple questions, but testing showed that it could answer only one part of a multi-step question. I eventually changed the planner to return a structured JSON plan containing every required computation, which Python then validates and executes.

I also ran into smaller issues that helped shape the final design. The planner initially generated the wrong argument structure for a what-if tool, and during another test the LLM incorrectly converted a model probability into a percentage. These problems reinforced the main design principle I ended up using: the LLM should understand the question and explain the result, while Python should handle data access, calculations, validation, and model execution.

I also learned a lot about turning notebook work into an actual application, including model serialization, modular code, Streamlit state, API secrets, Docker, deployment, and failure handling.

With more time, I would focus on improving model recall, making the analysis tools more flexible for complex segment questions, improving follow-up question context, and expanding what-if analysis to support multiple feature changes or entirely new hypothetical customer profiles.