import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.http import HttpTool
from autogen_core.models import UserMessage
from autogen_ext.models.ollama import OllamaChatCompletionClient
from typing import Any, Dict, List

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import HandoffTermination, TextMentionTermination
from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.teams import Swarm
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import json

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
open_router_api_key = os.getenv("OPENROUTER_API_KEY")


from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams

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


async def main():
    params = StdioServerParams(
        command="uvx", args=["mcp-server-time", "--local-timezone=America/New_York"]
    )

    model1 = OpenAIChatCompletionClient(model="gpt-4o")

    async with McpWorkbench(server_params=params) as workbench:
        agent = AssistantAgent(
            name="Time_Agent",
            system_message="You are helpful assistant",
            model_client=open_router_model_client,
            workbench=workbench,
            reflect_on_tool_use=True,
        )

    task = "what is the real time in Singapore"

    async for message in agent.run_stream(task=task):
        print("-" * 100)
        print(message)
        print("-" * 100)


if __name__ == "__main__":
    asyncio.run(main())
