
import os

from google import genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set.")

client = genai.Client(api_key=api_key)

print("Testing Gemini API...")

response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents="In two sentences, explain why astronauts exercise in space.",
)

print("\nRESPONSE:")
print(response.text)

print("\nUSAGE:")
print(response.usage_metadata)
