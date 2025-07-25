import os
from typing import Optional

import openai


SYSTEM_PROMPT = (
    "You are a personal log reflection assistant. Only respond based on the provided data."
)


def query_openai(prompt_text: str, model: str = "gpt-4") -> str:
    """Send the prompt to OpenAI's ChatCompletion API and return the response."""
    api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set")

    openai.api_key = api_key

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]

    response = openai.ChatCompletion.create(model=model, messages=messages)
    return response.choices[0].message["content"].strip()
