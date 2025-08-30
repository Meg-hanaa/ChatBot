import streamlit as st
import shelve
import uuid
import requests
import time
import pandas as pd
from db_chain import get_db_chain
import numpy as np
from langchain.memory import ConversationBufferMemory

# Constants
API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "mistralai/mixtral-8x7b-instruct"
USER_AVATAR = "👤"
BOT_AVATAR = "🤖"

# Load secrets
api_key = st.secrets["OPENROUTER_API_KEY"]
db_uri = st.secrets["DB_URI"]

# Page layout
st.set_page_config(page_title="🧠 Chatbot", page_icon="🤖")
st.title("🧠 Chatbot")

# Initialize database chain
chain = get_db_chain(db_uri, api_key)

# Migrate old sessions to new format
def migrate_old_sessions():
    with shelve.open("chat_history", writeback=True) as db:
        for key in list(db.keys()):
            if isinstance(db[key], list):
                db[key] = {"timestamp": None, "messages": db[key]}
migrate_old_sessions()

# Session management
def load_session(session_id):
    with shelve.open("chat_history") as db:
        data = db.get(session_id, {"timestamp": None, "messages": []})
        if isinstance(data, list):
            return {"timestamp": None, "messages": data}
        return data

def save_session(session_id, session_data):
    with shelve.open("chat_history") as db:
        db[session_id] = session_data

def delete_all_sessions():
    with shelve.open("chat_history") as db:
        db.clear()

def get_all_sessions():
    with shelve.open("chat_history") as db:
        return dict(db)

def create_new_session():
    new_id = str(uuid.uuid4())
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    session_data = {"timestamp": timestamp, "messages": []}
    save_session(new_id, session_data)
    st.session_state["session_id"] = new_id
    st.session_state["messages"] = []
    return new_id

# Initializes model choice and conversational memory buffer 
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = DEFAULT_MODEL

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
memory = st.session_state["memory"]

all_sessions_dict = get_all_sessions()
all_session_ids = list(all_sessions_dict.keys())

if "session_id" not in st.session_state or st.session_state["session_id"] not in all_session_ids:
    st.session_state["session_id"] = all_session_ids[-1] if all_session_ids else create_new_session()

session_data = load_session(st.session_state["session_id"])
st.session_state["messages"] = session_data["messages"]

# Sidebar
st.sidebar.title("💬 Sessions")

selected_session = st.sidebar.selectbox(
    "Choose a session:",
    all_session_ids,
    format_func=lambda x: (
        f"{all_sessions_dict[x]['timestamp']} ({x[:6]})"
        if isinstance(all_sessions_dict[x], dict) and "timestamp" in all_sessions_dict[x]
        else f"Session {x[:6]}"
    ),
    index=all_session_ids.index(st.session_state["session_id"]) if all_session_ids else 0
)

if selected_session and selected_session != st.session_state["session_id"]:
    st.session_state["session_id"] = selected_session
    st.session_state["messages"] = load_session(selected_session)["messages"]

if st.sidebar.button("➕ New Session"):
    create_new_session()
    st.rerun()

if st.sidebar.button("🗑 Delete All Sessions"):
    delete_all_sessions()
    st.sidebar.success("All sessions deleted.")
    st.rerun()


# Chat History with chart display
for idx, message in enumerate(st.session_state["messages"]):
    with st.chat_message(message["role"], avatar=USER_AVATAR if message["role"] == "user" else BOT_AVATAR):
        st.markdown(message["content"])
        # If this assistant message has a chart, display it
        if message.get("role") == "assistant" and "chart_df" in message and message["chart_df"] is not None and not message["chart_df"].empty:
            df = message["chart_df"].copy()
            if "Transparent" in df.columns:
                df["Transparent"] = df["Transparent"].astype(str).replace({"t": "Transparent", "f": "Non-Transparent"})
            # Always show chart type selector for every chartable message
            chart_type = st.selectbox(
                "Choose chart type",
                ["Bar", "Pie", "Line"],
                key=f"chart_type_{idx}",
                index=["Bar", "Pie", "Line"].index(message.get("chart_type", "Bar"))
            )
            # Update the chart_type in the message so it persists
            message["chart_type"] = chart_type
            # Determine which columns to use for the chart Special case: if 'num_parts' is present, use it
            if "num_parts" in df.columns:
                category_col = "set_num" if "set_num" in df.columns else df.columns[0]  # y  axis
                value_col = "num_parts"  # x axis
            else:
                category_col = None
                value_col = None
                for col in df.columns:
                    if category_col is None and (df[col].dtype == object or col.lower() == "year"):
                        category_col = col
                    if value_col is None and pd.api.types.is_numeric_dtype(df[col]) and col != category_col:
                        value_col = col
            if category_col is not None and value_col is not None:
                df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
                if chart_type == "Bar":
                    st.bar_chart(df.set_index(category_col)[value_col])
                elif chart_type == "Line":
                    st.line_chart(df.set_index(category_col)[value_col])
                elif chart_type == "Pie":
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots()
                    ax.pie(df[value_col], labels=df[category_col], autopct='%1.1f%%')
                    st.pyplot(fig)
            else:
                st.info("Not enough suitable columns for charting.")

# Display pending assistant message and chart (if any)
if "pending_assistant_message" in st.session_state:
    message = st.session_state["pending_assistant_message"]
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        st.markdown(message["content"])
        if "chart_df" in message and message["chart_df"] is not None and not message["chart_df"].empty:
            df = message["chart_df"].copy()
            if "Transparent" in df.columns:
                df["Transparent"] = df["Transparent"].astype(str).replace({"t": "Transparent", "f": "Non-Transparent"})
            st.dataframe(df)
            chart_type = st.selectbox(
                "Choose chart type",
                ["Bar", "Pie", "Line"],
                key="pending_chart_type",
                index=["Bar", "Pie", "Line"].index(message.get("chart_type", "Bar"))
            )
            message["chart_type"] = chart_type
            # Special case: if 'num_parts' is present, use it
            if "num_parts" in df.columns:
                category_col = "set_num" if "set_num" in df.columns else df.columns[0]
                value_col = "num_parts"
            else:
                category_col = None
                value_col = None
                for col in df.columns:
                    if category_col is None and (df[col].dtype == object or col.lower() == "year"):
                        category_col = col
                    if value_col is None and pd.api.types.is_numeric_dtype(df[col]) and col != category_col:
                        value_col = col
            if category_col is not None and value_col is not None:
                df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
                if chart_type == "Bar":
                    st.bar_chart(df.set_index(category_col)[value_col])
                elif chart_type == "Line":
                    st.line_chart(df.set_index(category_col)[value_col])
                elif chart_type == "Pie":
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots()
                    ax.pie(df[value_col], labels=df[category_col], autopct='%1.1f%%')
                    st.pyplot(fig)
            else:
                st.info("Not enough suitable columns for charting.")
# After displaying, remove the pending message so it doesn't show again
if "pending_assistant_message" in st.session_state:
    del st.session_state["pending_assistant_message"]

# Chat Input
def is_chartable_query(query):
    chart_keywords = [
        "most popular", "per year", "distribution", "top", "frequency", "proportion",
        "usage", "by year", "group by", "vs", "number of", "count of", "how many", "trend"
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in chart_keywords)

prompt = st.chat_input("Say something...")
is_new_prompt = prompt is not None and prompt != ""

if prompt is not None and prompt != "":
    # Append user message to chat history
    st.session_state["messages"].append({"role": "user", "content": prompt})

    # Save user message to memory (output will be filled after LLM response)
    memory.save_context({"input": prompt}, {"output": ""})

    # Build chat history for context
    chat_history = memory.load_memory_variables({})["chat_history"]

    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        placeholder = st.empty()
        with st.spinner("🚀 Thinking..."):
            try:
                sql_response = chain.run(prompt, chat_history=chat_history)
                if isinstance(sql_response, tuple) and len(sql_response) == 2:
                    markdown_str, df = sql_response
                    # Apply mapping BEFORE storing in session state
                    if "Transparent" in df.columns:
                        df["Transparent"] = df["Transparent"].astype(str).replace({"t": "Transparent", "f": "Non-Transparent"})
                    placeholder.markdown(markdown_str, unsafe_allow_html=True)
                    if is_chartable_query(prompt):
                        # Show the chart for the current response immediately
                        chart_type = "Bar"
                        if "num_parts" in df.columns:
                            category_col = "set_num" if "set_num" in df.columns else df.columns[0]
                            value_col = "num_parts"
                        else:
                            category_col = None
                            value_col = None
                            for col in df.columns:
                                if category_col is None and (df[col].dtype == object or col.lower() == "year"):
                                    category_col = col
                                if value_col is None and pd.api.types.is_numeric_dtype(df[col]) and col != category_col:
                                    value_col = col
                        if category_col is not None and value_col is not None:
                            df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
                            st.bar_chart(df.set_index(category_col)[value_col])
                        # Store the message as pending so it is shown in chat history immediately
                        st.session_state["pending_assistant_message"] = {
                            "role": "assistant",
                            "content": markdown_str,
                            "chart_df": df,
                            "chart_type": chart_type,
                            "chart_query": prompt
                        }
                    else:
                        st.session_state["pending_assistant_message"] = {
                            "role": "assistant",
                            "content": markdown_str
                        }
                    memory.save_context({"input": prompt}, {"output": markdown_str})
                else:
                    placeholder.markdown(sql_response)
                    memory.save_context({"input": prompt}, {"output": sql_response})
                    st.session_state["pending_assistant_message"] = {
                        "role": "assistant",
                        "content": str(sql_response)
                    }
            except Exception as e:
                placeholder.markdown("⚠ DB Error.")
                st.error(str(e))

    # After generating the pending assistant message, immediately move it to chat history
    if "pending_assistant_message" in st.session_state:
        st.session_state["messages"].append(st.session_state["pending_assistant_message"])
        del st.session_state["pending_assistant_message"]
        if not is_new_prompt:
            st.rerun()

    # Now save chat history
    current_session_id = st.session_state["session_id"]
    session_data = load_session(current_session_id)
    session_data["messages"] = st.session_state["messages"]
    save_session(current_session_id, session_data)

    