from google.adk.agents import Agent
from dotenv import load_dotenv
from google.adk.models.lite_llm import LiteLlm
import os


load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")


OPENROUTER_API_KEY = (
    "sk"
)

model1 = LiteLlm(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
model1
# openrouter_model = LiteLlm(
#     model="deepseek/deepseek-r1-0528-qwen3-8b:free",
#     api_key=OPENROUTER_API_KEY,
# )

root_agent = Agent(
    model=model1,
    name="root_agent",
    description="A helpful assistant for user questions.",
    instruction="Answer user questions to the best of your knowledge",
)
