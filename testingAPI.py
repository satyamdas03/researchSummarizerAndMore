import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

def ai_chatbot(query):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    query = "What are the key findings in recent studies about deep learning?"
    print(ai_chatbot(query))
