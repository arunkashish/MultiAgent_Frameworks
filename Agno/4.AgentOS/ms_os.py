# wstart fast api server on the port ussing command : -->uvicorn ms_os:app --reload --port 7777
# go to os.agno.com
# login and connect fast api server
# check SWAGGER --> http://localhost:7777/docs

# ms_os.py
OPENROUTER_API_KEY = "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"  # Replace with actual key # Replace with your actual key # Replace with your actual key
# ms_os.py
# ms_os.py
import os
from datetime import datetime
from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from agno.os import AgentOS
from agno.db.sqlite import SqliteDb
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session

# ============================
# Configuration
# ============================
# OPENROUTER_API_KEY = "YOUR_OPENROUTER_KEY"  # Replace with your actual key
DB_PATH = "./temp/agent_messages.db"
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# ============================
# SQLAlchemy setup
# ============================
engine = create_engine(
    f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False}
)
Base = declarative_base()
SessionLocal = scoped_session(sessionmaker(bind=engine))


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, index=True)
    role = Column(String)
    message = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)


# Create tables
Base.metadata.create_all(engine)


# ============================
# Helper functions for memory
# ============================
def save_message(user_id: str, role: str, message: str):
    session = SessionLocal()
    msg = Message(user_id=user_id, role=role, message=message)
    session.add(msg)
    session.commit()
    session.close()
    print(f"[Saved] {role}: {message}")


def get_conversation_history(user_id: str):
    session = SessionLocal()
    rows = session.query(Message).filter_by(user_id=user_id).order_by(Message.id).all()
    session.close()
    return [{"role": row.role, "content": row.message} for row in rows]


# ============================
# Setup Agno Agent
# ============================
agent_db = SqliteDb(db_file=DB_PATH)  # Only for internal use by Agno

assistant = Agent(
    name="Assistant",
    model=OpenRouter(
        api_key=OPENROUTER_API_KEY,
        models=["deepseek/deepseek-r1-0528:free"],
    ),
    instructions=["You are a helpful AI assistant."],
    markdown=True,
    db=agent_db,
)

# ============================
# Wrap run() to include SQLAlchemy memory
# ============================
original_run = assistant.run


def run_with_memory(*args, **kwargs):
    user_id = kwargs.get("user_id", "default_user")

    # Get user message
    if "message" in kwargs:
        user_message = kwargs["message"]
    elif len(args) > 0:
        user_message = args[0]
    else:
        user_message = ""

    # Load conversation history
    history = get_conversation_history(user_id)
    history.append({"role": "user", "content": user_message})

    # Generate assistant response
    # Some OpenRouter models may use .chat() instead of .run()
    response = assistant.model.run(messages=history)

    # Save messages to SQLAlchemy DB
    save_message(user_id, "user", user_message)
    save_message(user_id, "assistant", response)

    return response


assistant.run = run_with_memory

# ============================
# Define AgentOS
# ============================
agent_os = AgentOS(
    id="my-first-os",
    description="AgentOS with persistent SQLAlchemy memory",
    agents=[assistant],
)

# ============================
# FastAPI app
# ============================
app = agent_os.get_app()

# ============================
# Run server
# ============================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ms_os:app", host="0.0.0.0", port=7777, reload=True)
