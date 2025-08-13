import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from autogen_ext.models.openai import OpenAIChatCompletionClient
import requests


class CustomLLMClient:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip("/")
        self.model_info = {
            "family": "custom_llm",
            "vision": False,
            "function_calling": False,
            "json_output": False,
        }

    def chat(self, prompt, max_tokens=100):
        payload = {"prompt": prompt, "max_tokens": max_tokens}
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(self.base_url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"Error calling custom LLM API: {e}")


def setup_llm_clients():
    load_dotenv()

    groq_api_key = os.getenv("GROQ_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    open_router_api_key = os.getenv("OPENROUTER_API_KEY")

    custom_llm_api_url = os.getenv("CUSTOM_LLM_API_URL", "http://192.168.10.8:1234")

    if not groq_api_key or not openai_api_key:
        print("Warning: One or more API keys are missing. Please check your .env file.")
        return None, None, None, None, None

    print("LLM API keys loaded successfully.")

    chat_groq = ChatGroq(model="qwen-qwq-32B", groq_api_key=groq_api_key)
    chat_openai = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)
    model1 = OpenAIChatCompletionClient(model="gpt-4o", openai_api_key=openai_api_key)
    open_router_model_client = OpenAIChatCompletionClient(
        base_url="https://openrouter.ai/api/v1",
        model="nvidia/llama-3.1-nemotron-70b-instruct",
        api_key=open_router_api_key,
        model_info={
            "family": "deepseek",
            "vision": True,
            "function_calling": True,
            "json_output": False,
        },
    )

    custom_llm_client = CustomLLMClient(base_url=custom_llm_api_url)

    return chat_groq, chat_openai, model1, open_router_model_client, custom_llm_client
