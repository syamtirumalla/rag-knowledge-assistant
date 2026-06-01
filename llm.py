import os
from dotenv import load_dotenv
from groq import Groq as GroqClient
from llama_index.core import Settings
from llama_index.llms.groq import Groq

load_dotenv()

def setup_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env file!")
    
    llm = Groq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.1,
        max_tokens=1024,
    )
    Settings.llm = llm
    return llm

if __name__ == "__main__":
    # Direct Groq API test (bypasses llama-index)
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    client = GroqClient(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "What is RAG in AI? Answer in 2 sentences."}]
    )
    print("LLM Test:", response.choices[0].message.content)