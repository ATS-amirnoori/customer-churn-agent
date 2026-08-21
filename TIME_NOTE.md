# Development Time Note

I spent approximately **10–12 focused hours** on this project.

This was slightly above the suggested 8–10 hour range, mostly because I worked through the project very methodically. I wanted to understand the reasoning behind each part of the implementation instead of only getting something working as quickly as possible, especially once I reached the agent and LLM portions that were newer to me.

A rough breakdown of my time was:

- **Data cleaning, EDA, and understanding the dataset:** ~2 hours
- **Model training, evaluation, and comparison:** ~1.5–2 hours
- **Moving the model and analysis into reusable local Python tools:** ~1.5 hours
- **Groq integration, agent planning, tool execution, and verification:** ~3 hours
- **Streamlit interface and application testing:** ~1 hour
- **Dockerization, deployment, debugging, and final documentation:** ~1.5–2 hours

The agent workflow took the most time because I ran into issues that only became visible through testing, especially around multi-step questions, structured planning, tool argument formats, and keeping numerical answers grounded in Python calculations.

I stopped after completing a working hosted Streamlit application with the trained model, structured agent planning, verified Python tool execution, what-if predictions, Docker support, and project documentation.

With additional time, I would focus on improving model recall, making the analysis tools more flexible for complex segment questions, improving follow-up conversation context, and expanding the what-if functionality.