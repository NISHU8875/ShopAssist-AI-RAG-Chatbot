import os
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------

chitchat_system_prompt = """You are a friendly and helpful e-commerce shopping assistant. You can:

1. Have casual conversations and greet users warmly
2. Provide fashion advice and styling suggestions
3. Offer wellness and lifestyle tips
4. Share general information like date and time
5. Help users feel comfortable and engaged

Guidelines:
- Be warm, friendly, and conversational
- Keep responses concise (2–4 sentences)
- For fashion advice, consider occasions, seasons, and personal style
- For wellness, give general healthy lifestyle tips
- Always maintain a helpful shopping assistant persona
- If asked about specific products, gently remind users they can ask about product searches

Remember: You're part of an e-commerce platform, so stay relevant to shopping and lifestyle when possible.
"""

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def get_current_datetime_info():
    now = datetime.now()
    return {
        "date": now.strftime("%A, %B %d, %Y"),
        "time": now.strftime("%I:%M %p"),
        "day": now.strftime("%A"),
    }

# ---------------------------------------------------------------------
# Chitchat Chain
# ---------------------------------------------------------------------

def chitchat_chain(query: str) -> str:
    datetime_info = get_current_datetime_info()
    context_note = (
        f"\n\nCurrent date and time: "
        f"{datetime_info['date']}, {datetime_info['time']}"
    )

    messages = [
        {"role": "system", "content": chitchat_system_prompt + context_note},
        {"role": "user", "content": query},
    ]

    try:
        response = client.responses.create(
            model="gpt-5-mini",
            input=messages
        )
        return response.output_text

    except Exception as e:
        return f"I'm sorry, I ran into an error. Please try again."

# ---------------------------------------------------------------------
# Local Testing
# ---------------------------------------------------------------------

if __name__ == "__main__":
    test_queries = [
        "Hi, I'm Pankaj!",
        "What should I wear for a summer wedding?",
        "What's today's date?",
        "Can you give me some wellness tips?",
        "What are the latest fashion trends?",
    ]

    for query in test_queries:
        print("\nQuery:", query)
        print("Answer:", chitchat_chain(query))
        print("-" * 80)
