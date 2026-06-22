import anthropic
import os
import dotenv

dotenv.load_dotenv()  # Load environment variables from .env file

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Say hello in five words."}]
)

print(response.content[0].text)