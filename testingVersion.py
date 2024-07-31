import os
from pprint import pprint
import requests
from dotenv import load_dotenv
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableLambda
from langchain.schema.runnable.config import RunnableConfig
from langchain.memory import ConversationBufferMemory
from chainlit.types import ThreadDict
from typing import Optional
import chainlit as cl
import bcrypt
import mysql.connector
from openai import AsyncOpenAI

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
API_KEY = os.getenv("API_KEY")
BASE_URL = "https://api.deepbricks.ai/v1/"
client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
settings = {
    "model": "llama-3.1-405b",
    "temperature": 0,
    "max_tokens": 4095,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}


# Function to search using Bing Search API
def search(query):
    mkt = 'zh-CN'
    params = {'q': query, 'mkt': mkt, 'freshness': 'Month'}
    headers = {'Ocp-Apim-Subscription-Key': subscription_key}

    try:
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()
        return response.json()['webPages']['value']
    except Exception as ex:
        raise ex


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


# Setup runnable model
def setup_runnable():
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    model = ChatOpenAI(streaming=True)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Answer every dialogue in Chinese."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    runnable = (
            RunnablePassthrough.assign(
                history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
            )
            | prompt
            | model
            | StrOutputParser()
    )
    cl.user_session.set("runnable", runnable)


# User authentication callback
@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
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
    setup_runnable()
    await cl.Message(content="我是基于" + settings.get("model") + "的API接口的聊天机器人，请随时向我提问 :)").send()


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
    setup_runnable()


# Message handling callback
@cl.on_message
async def on_message(message: cl.Message):
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    message_history = cl.user_session.get("message_history", [])

    message_history.append({"role": "user", "content": message.content})
    cl.user_session.set("message_history", message_history)

    try:
        # Bing search API call
        search_results = search(message.content)
        search_prompts = [
            f"来源:\n标题: {result['name']}\n网址: {result['url']}\n内容: {result['snippet']}\n" for result in
            search_results
        ]
        search_content = "Use the following sources to answer the question:\n\n".join(
            search_prompts) + "\n\nQuestion: " + message.content + "\n\n"

        # Sending the search results to the user
        await cl.Message(content=search_content).send()

        # Add search results to the message history
        message_history.append({"role": "system", "content": search_content})
        cl.user_session.set("message_history", message_history)

        # Sending message to Llama 3.1
        msg = cl.Message(content="")
        await msg.send()

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
