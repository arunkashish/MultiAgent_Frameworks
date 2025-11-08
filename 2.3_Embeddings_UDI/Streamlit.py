import streamlit as st
from langchain.schema import HumanMessage, AIMessage
import import_ipynb
import check  # your notebook
import json
import re

workflow = check.workflow
config = check.config

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="LangGraph Multi-Tool Demo", page_icon="🤖", layout="wide"
)

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.title("⚙️ Settings")
    st.info("This demo showcases a LangGraph-powered multi-tool agent.")
    st.markdown("**Available Tools:**")
    st.markdown(
        "- 🌐 Web Search (DuckDuckGo)\n"
        "- ⏰ Current Time\n"
        "- 📈 Stock Prices\n"
        "- 👤 User Details\n"
        "- 💬 Normal Chat"
    )

# -------------------- MAIN TITLE --------------------
st.title("🤖 LangGraph Multi-Tool Demo")
st.caption(
    "Ask me about web search, current time, stock prices, user details, or just chat normally."
)

# -------------------- SESSION STATE --------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------- USER INPUT --------------------
with st.container():
    with st.form(key="user_input_form", clear_on_submit=True):
        user_query = st.text_input(
            "💬 Enter your query:", placeholder="e.g., What's the stock price of Apple?"
        )
        submit_button = st.form_submit_button(label="🚀 Send")

# -------------------- PROCESS QUERY --------------------
if submit_button and user_query:
    st.session_state.history.append(HumanMessage(content=user_query))

    try:
        # Call workflow
        result = workflow.invoke({"messages": st.session_state.history}, config=config)
        last_msg = result["messages"][-1].content

        # ---------------- DETECT TOOL ----------------
        tool_used = "General 🤔"
        if last_msg.startswith("[Tool: DuckDuckGo]"):
            tool_used = "🌐 Web Search"
        elif last_msg.startswith("[Tool: Time]"):
            tool_used = "⏰ Current Time"
        elif last_msg.startswith("[Tool: Stock]"):
            tool_used = "📈 Stock Price"
        elif last_msg.startswith("[Tool: UserDetails]"):
            tool_used = "👤 User Details"
        elif last_msg.startswith("[Tool: Normal Chat]"):
            tool_used = "💬 Normal Chat"

        # ---------------- DISPLAY OUTPUT ----------------
        st.success(f"**Tool Used:** {tool_used}")
        st.write("### 🤖 Assistant")

        # Remove prefix for clean display
        clean_msg = re.sub(r"^\[Tool:[^\]]+\]\s*", "", last_msg).strip()

        # Try JSON formatting
        try:
            parsed = json.loads(clean_msg)
            pretty_json = json.dumps(parsed, indent=2)
            st.code(pretty_json, language="json")
        except:
            if ":" in clean_msg:
                # Key-value style formatting
                lines = re.split(r"\s(?=\w+:)", clean_msg)
                formatted = "\n".join(lines)
                st.code(formatted, language="yaml")
            else:
                st.markdown(f"> {clean_msg}")

        # Append AI reply to history
        st.session_state.history.append(AIMessage(content=last_msg))

    except Exception as e:
        st.error(f"❌ Error: {e}")

# -------------------- CHAT HISTORY --------------------
st.markdown("---")
with st.expander("📜 Conversation History"):
    for msg in st.session_state.history:
        if isinstance(msg, HumanMessage):
            st.markdown(f"**🧑 You:** {msg.content}")
        else:
            st.markdown(f"**🤖 Assistant:** {msg.content}")
