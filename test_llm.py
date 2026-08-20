# Verify that the local application can securely connect to the Groq LLM API.

import os
from dotenv import load_dotenv
from groq import Groq


# Load variables stored in the local .env file
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY was not found in the .env file.")


# Create a Groq client using the private API key
client = Groq(api_key=api_key)

# Send one small request to confirm the LLM connection works
response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "user",
            "content": "Reply with exactly: Groq connection successful."
        }
    ],
    temperature=0
)


print(response.choices[0].message.content)