import os
from langchain.memory import ConversationBufferMemory
from chainlit.types import ThreadDict
from typing import Optional
import chainlit as cl
import bcrypt
import mysql.connector
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# DB configuration
DB_CONFIG = {
    'user': 'root',
    'password': os.getenv("DB_PWD"),
    'host': '127.0.0.1',
    'database': 'user_db',
}

# Bing Search API configuration
subscription_key = os.getenv('BING_SEARCH_V7_SUBSCRIPTION_KEY')
endpoint = os.getenv('BING_SEARCH_V7_ENDPOINT') + "v7.0/search"

# API client configuration for Llama 3.1
API_KEY = os.getenv("NV_API_KEY")
BASE_URL = "https://integrate.api.nvidia.com/v1"
client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
settings = {
    "model": "deepseek-ai/deepseek-r1",
    "temperature": 0,
    "max_tokens": 4096,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}


# Database connection and user management functions
def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)


def read_user(username):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT password_hash FROM users WHERE username = %s", (username,))
    user = cursor.fetchone()
    conn.close()
    return user


def write_user(username, password_hash):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s)", (username, password_hash))
    conn.commit()
    conn.close()


@cl.set_chat_profiles
async def chat_profile(current_user: cl.User):
    models = [
        "deepseek-ai/deepseek-r1"
    ]
    model_icon_map = {
        "deepseek-ai/deepseek-r1": "https://registry.npmmirror.com/@lobehub/icons-static-png/latest/files/dark/deepseek-color.png"
    }
    return [
        cl.ChatProfile(
            name=models[0],
            markdown_description="**DeepSeek-R1** achieves performance comparable to OpenAI-o1 across math, code, and reasoning tasks. To support the research community, we have open-sourced DeepSeek-R1-Zero, DeepSeek-R1, and six dense models distilled from DeepSeek-R1 based on Llama and Qwen. DeepSeek-R1-Distill-Qwen-32B outperforms OpenAI-o1-mini across various benchmarks, achieving new state-of-the-art results for dense models.",
            icon=model_icon_map.get(models[0], "https://www.shutterstock.com/image-vector/letter-llm-logo-template-vector-260nw-1673993428.jpg"),
            starters=[
                cl.Starter(
                    label="Code Debug",
                    message="Help me debug the following code",
                    icon="/public/proofread.jpg",
                ),
                cl.Starter(
                    label="Code Syntax",
                    message="Give me the general syntax of the SML language",
                    icon="/public/content.svg",
                ),
            ]
        )
    ]


# User authentication callback
@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    return cl.User(identifier=username, metadata={"role": "user", "provider": "bypass"})
    
    user = read_user(username)
    if user and bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
        return cl.User(identifier=username, metadata={"role": "user", "provider": "credentials"})
    elif not user:
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        write_user(username, password_hash)
        return cl.User(identifier=username, metadata={"role": "user", "provider": "credentials"})
    else:
        return None


# Chat start callback
@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))


# Chat resume callback
@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    memory = ConversationBufferMemory(return_messages=True)
    root_messages = [m for m in thread["steps"] if m["parentId"] is None]
    for message in root_messages:
        if message["type"] == "user_message":
            memory.chat_memory.add_user_message(message["output"])
        else:
            memory.chat_memory.add_ai_message(message["output"])
    cl.user_session.set("memory", memory)
    await cl.Message("").send()


# Message handling callback
@cl.on_message
async def on_message(message: cl.Message):
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    message_history = cl.user_session.get("message_history", [])
    current_profile = cl.user_session.get("chat_profile")
    message_history.append({"role": "user", "content": message.content})
    cl.user_session.set("message_history", message_history)

    try:
        msg = cl.Message(content="")
        await msg.send()

        settings["model"] = current_profile
        stream = await client.chat.completions.create(
            messages=message_history, stream=True, **settings
        )

        async for part in stream:
            if token := part.choices[0].delta.content or "":
                await msg.stream_token(token)

        message_history.append({"role": "assistant", "content": msg.content})
        await msg.update()

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        await cl.Message(content=error_message).send()

    # Update memory with chat history
    memory.chat_memory.add_user_message(message.content)
    memory.chat_memory.add_ai_message(msg.content)

    # Save session state
    cl.user_session.set("memory", memory)
    cl.user_session.set("message_history", message_history)
