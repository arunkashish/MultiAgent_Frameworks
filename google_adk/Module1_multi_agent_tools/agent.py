import os
from dotenv import load_dotenv
from pprint import pprint
import datetime
from zoneinfo import ZoneInfo
from openai import OpenAI
from google.adk import Agent
from google.adk.models import BaseLlm, LlmResponse
from pydantic import PrivateAttr
from typing import ClassVar, Set

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")


# -----------------------------
# OpenAI LLM wrapper for ADK
# -----------------------------
class OpenAILlm(BaseLlm):
    model: str
    _client: OpenAI = PrivateAttr()
    VALID_ROLES: ClassVar[set] = {
        "function",
        "user",
        "tool",
        "system",
        "developer",
        "assistant",
    }

    def __init__(self, model_name="gpt-4o-mini"):
        super().__init__(model=model_name)
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def generate_content_async(self, messages, **kwargs):
        formatted_messages = []
        for m in messages:
            if isinstance(m, dict):
                role = m.get("role", "user").lower()
                if role not in self.VALID_ROLES:
                    role = "user"
                formatted_messages.append({"role": role, "content": str(m["content"])})
            elif isinstance(m, tuple) and len(m) == 2:
                role = str(m[0]).lower()
                if role not in self.VALID_ROLES:
                    role = "user"
                formatted_messages.append({"role": role, "content": str(m[1])})
            else:
                raise ValueError(f"Invalid message format: {m}")

        response = self._client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            temperature=kwargs.get("temperature", 0),
            stream=False,
        )

        content_text = response.choices[0].message.content
        content_str = (
            "".join(str(c) for c in content_text)
            if isinstance(content_text, list)
            else str(content_text)
        )

        # Wrap in LlmResponse with proper 'content' dict
        yield LlmResponse(content={"text": content_str}, usage_metadata=None)


# -----------------------------
# Tool functions
# -----------------------------
def get_weather(city: str) -> dict:
    if city.lower() == "new york":
        return {
            "status": "success",
            "report": "The weather in New York is sunny with a temperature of 25°C (77°F).",
        }
    return {
        "status": "error",
        "error_message": f"Weather information for '{city}' is not available.",
    }


def get_current_time(city: str) -> dict:
    if city.lower() == "new york":
        tz = ZoneInfo("America/New_York")
        now = datetime.datetime.now(tz)
        report = (
            f"The current time in {city} is {now.strftime('%Y-%m-%d %H:%M:%S %Z%z')}"
        )
        return {"status": "success", "report": report}
    return {
        "status": "error",
        "error_message": f"Sorry, I don't have timezone information for {city}.",
    }


# -----------------------------
# Root agent
# -----------------------------
root_agent = Agent(
    name="weather_time_agent",
    model=OpenAILlm(model_name="gpt-4o-mini"),
    description="Agent to answer questions about the time and weather in a city.",
    instruction="You are a helpful agent who can answer user questions about the time and weather in a city.",
    tools=[get_weather, get_current_time],
)


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    pprint(root_agent.__dict__)
