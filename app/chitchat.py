import os
from phi.model.openai import OpenAIChat
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Initialize OpenAI model (API key auto-loaded from env)
model = OpenAIChat(id="gpt-5-mini")

# groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))

chitchat_system_prompt = """You are a friendly and helpful e-commerce shopping assistant. You can:

1. Have casual conversations and greet users warmly
2. Provide fashion advice and styling suggestions
3. Offer wellness and lifestyle tips
4. Share general information like date, time, and weather advice
5. Help users feel comfortable and engaged

Guidelines:
- Be warm, friendly, and conversational
- Keep responses concise (2-4 sentences typically)
- For fashion advice, consider occasions, seasons, and personal style
- For wellness, give general healthy lifestyle tips
- Always maintain a helpful shopping assistant persona
- If asked about specific products, gently remind users they can ask about product searches

Remember: You're part of an e-commerce platform, so stay relevant to shopping and lifestyle when possible."""


def get_current_datetime_info():
    """Get current date and time information"""
    now = datetime.now()
    return {
        'date': now.strftime('%A, %B %d, %Y'),
        'time': now.strftime('%I:%M %p'),
        'day': now.strftime('%A'),
    }


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
        response = model.run(messages)
        return response.content

    except Exception as e:
        return f"I'm sorry, I ran into an error: {e}"


if __name__ == '__main__':
    # Test cases
    test_queries = [
        "Hi, I'm Pankaj!",
        "What should I wear for a summer wedding?",
        "What's today's date?",
        "Can you give me some wellness tips?",
        "What are the latest fashion trends?",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        answer = chitchat_chain(query)
        print(f"Answer: {answer}")
        print("-" * 80)