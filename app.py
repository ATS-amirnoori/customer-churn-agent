# Chat interface for the customer churn analysis agent.

# Streamlit reruns this script after each interaction, so session_state stores
# the current conversation between reruns. Previous messages are redrawn first,
# then a new submitted question is sent through the verified agent workflow
# and both the user question and assistant answer are saved for the next rerun.


# Save tool and verification metadata with assistant messages so users can
# optionally inspect how each answer was computed without cluttering the chat.

import traceback
import streamlit as st

from src.agent import answer_question


st.set_page_config(
    page_title="Customer Churn Analysis Agent",
    page_icon="📊"
)

# Provide a small control panel without cluttering the main chat interface.
with st.sidebar:

    st.header("About")

    st.write(
        "This agent answers questions using the customer churn dataset "
        "and a trained churn prediction model."
    )

    st.write(
        "Responses are grounded in approved Python analysis tools and "
        "verified before being returned."
    )

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

st.title("Customer Churn Analysis Agent")

st.write(
    "Ask questions about the customer dataset or request churn-risk predictions."
)


# Store the visible conversation so messages remain on screen across Streamlit reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []


# Redisplay previous messages and any saved analysis details.
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Only assistant messages with saved tool information show this section.
        if "tool_results" in message:

            with st.expander("Analysis details"):

                for tool_result in message["tool_results"]:

                    st.write(f"**Tool:** `{tool_result['tool']}`")

                    st.write("**Arguments:**")
                    st.json(tool_result["arguments"])

                st.write(
                    f"**Verification:** {message['verification']}"
                )


# Display the chat input at the bottom of the page.
user_question = st.chat_input(
    "Ask about customer churn..."
)


if user_question:

    # Save the user's message so it remains visible after future reruns.
    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })

    # Display the user's message immediately.
    with st.chat_message("user"):
        st.markdown(user_question)


    # Run the existing verified agent workflow and display the response.
    with st.chat_message("assistant"):

        try:
            with st.spinner("Analyzing..."):
                result = answer_question(user_question)

            st.markdown(result["answer"])

            with st.expander("Analysis details"):

                for tool_result in result["tool_results"]:

                    st.write(f"**Tool:** `{tool_result['tool']}`")

                    st.write("**Arguments:**")
                    st.json(tool_result["arguments"])

                st.write(
                    f"**Verification:** {result['verification']}"
                )

            assistant_message = {
                "role": "assistant",
                "content": result["answer"],
                "tool_results": result["tool_results"],
                "verification": result["verification"]
            }
            
        # Catch unexpected application or API failures so the Streamlit interface
        # remains usable instead of exposing a raw traceback to the user.
        except Exception as error:

            # Print the full traceback to the server logs for debugging while keeping
            # the user-facing interface clean.
            print("\nUNEXPECTED APPLICATION ERROR:")
            traceback.print_exc()

            error_message = (
                "Sorry, I couldn't process that question because an unexpected "
                "application error occurred."
            )

            st.error(error_message)

            assistant_message = {
                "role": "assistant",
                "content": error_message
            }


    # Save the answer and its analysis metadata so both can be
    # reconstructed when Streamlit reruns the script.
    st.session_state.messages.append(assistant_message)