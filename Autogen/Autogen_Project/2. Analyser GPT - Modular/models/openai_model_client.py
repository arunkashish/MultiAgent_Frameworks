from autogen_ext.models.openai import OpenAIChatCompletionClient
from config.constants import MODEL_OPENAI
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
open_router_api_key = (
    "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"
)


def get_model_client():
    openai_model_client = OpenAIChatCompletionClient(
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
    return openai_model_client
