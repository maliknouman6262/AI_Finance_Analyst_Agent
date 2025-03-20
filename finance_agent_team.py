import os
import streamlit as st
import json
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.storage.agent.sqlite import SqliteAgentStorage
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools

# Streamlit UI
st.set_page_config(page_title="AI Multi-Agent System", layout="wide")
st.sidebar.title("⚙️ Chat Controls")

# Get API Key from user
openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
if not openai_api_key:
    st.sidebar.warning("Please enter your OpenAI API key to proceed.")
    st.stop()

# Initialize Agents
web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=OpenAIChat(id="gpt-4o", api_key=openai_api_key),
    tools=[DuckDuckGoTools()],
    storage=SqliteAgentStorage(table_name="web_agent", db_file="agents.db"),
    add_history_to_messages=True,
    markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=OpenAIChat(id="gpt-4o", api_key=openai_api_key),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True, company_news=True)],
    instructions=["Always use tables to display data"],
    storage=SqliteAgentStorage(table_name="finance_agent", db_file="agents.db"),
    add_history_to_messages=True,
    markdown=True,
)

research_agent = Agent(
    name="Research Agent",
    role="Conduct in-depth research on various topics",
    model=OpenAIChat(id="gpt-4o", api_key=openai_api_key),
    tools=[DuckDuckGoTools()],
    storage=SqliteAgentStorage(table_name="research_agent", db_file="agents.db"),
    add_history_to_messages=True,
    markdown=True,
)

simple_agent = Agent(
    name="Simple Agent",
    role="Provide general AI-powered responses",
    model=OpenAIChat(id="gpt-4o", api_key=openai_api_key),
    storage=SqliteAgentStorage(table_name="simple_agent", db_file="agents.db"),
    add_history_to_messages=True,
    markdown=True,
)

# Sidebar Controls
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

if st.sidebar.button("💾 Save Chat"):
    chat_text = "\n\n".join([f"You: {chat['query']}\nAI: {chat['response']}" for chat in st.session_state.chat_history])
    st.sidebar.download_button("Download Chat History", chat_text, "chat_history.txt", "text/plain")

st.title("🤖 AI Multi-Agent System")

# Select Agent
agent_options = {
    "🌍 Web Search Agent": web_agent,
    "📈 Finance Agent": finance_agent,
    "📚 Research Agent": research_agent,
    "💬 Simple Agent": simple_agent
}
selected_agent_name = st.selectbox("Select an AI Agent:", list(agent_options.keys()))
selected_agent = agent_options[selected_agent_name]

# Initialize Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User Input
query = st.text_input("🔍 **Enter your query:**")

if st.button("Run Agent"):
    if query:
        with st.spinner(f"⏳ {selected_agent.name} is processing..."):
            response = selected_agent.run(query)
            clean_response = response.content if hasattr(response, "content") else "⚠️ No valid response received."
            
            # Save response to chat history
            st.session_state.chat_history.append({"query": query, "response": clean_response})
            
            # Display response
            st.markdown(f"### 📢 AI Response from {selected_agent.name}:")
            st.markdown(clean_response, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a query before clicking 'Run Agent'.")

# Display Chat History
if st.session_state.chat_history:
    st.subheader("📝 Chat History")
    for chat in st.session_state.chat_history:
        st.markdown(f"**You:** {chat['query']}")
        st.markdown(f"**AI:** {chat['response']}")
        st.markdown("---")
