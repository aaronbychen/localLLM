import os
from pprint import pprint
import requests
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
API_KEY = os.getenv("API_KEY")
BASE_URL = "https://api.deepbricks.ai/v1/"
client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
settings = {
    "model": "claude-3.5-sonnet",
    "temperature": 0,
    "max_tokens": 4095,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}     


# Function to search using Bing Search API
def search(query):
    mkt = 'zh-CN'
    params = {'q': query, 'mkt': mkt}
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
            ("system", "You are a extremely helpful chatbot. Answer every dialogue in Chinese. Your name is 小驼"),
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


@cl.set_chat_profiles
async def chat_profile(current_user: cl.User):
    return [
        cl.ChatProfile(
            name=settings.get("model"),
            markdown_description="底层LLM模型为**llama-3.1**，有**4050亿**级参数",
            icon="https://www.shutterstock.com/image-vector/letter-llm-logo-template-vector-260nw-1673993428.jpg",
            starters=[
                cl.Starter(
                    label="校对文字",
                    message="帮我校对下面这段文字，标粗修改处，以及列出哪里被修改了：“近日，一家公司发布了一款全新智能手机，该手机配被了最新的处理器和高像素摄像头，此外还具备长效电池和防水功能。据城，这款手机将在本月底正式上市销售。然而，用户需要注意的是，由于供货量有限，可能会出现依时断货的情况。此手机在市场上的定价预计会非常具有竞征力，是消费者的一个理想选择。”",
                    icon="/public/proofread.jpg",
                ),
                cl.Starter(
                    label="内容创作",
                    message="给我写一段300字的示例新闻稿，内容不限，要求文笔清晰，有逻辑条理",
                    icon="/public/content.svg",
                ),
            ]
        ),
        cl.ChatProfile(
            name=settings.get("model") + " & Bing",
            markdown_description="底层LLM模型为**llama-3.1**，有**4050亿**级参数，集成了**必应**搜索，消耗资源稍多，暂不支持上下文（试用）",
            icon="https://logos-world.net/wp-content/uploads/2021/02/Bing-Emblem.png",
            starters=[
                cl.Starter(
                    label="剩余天数",
                    message="2024年还剩下多少天？",
                    icon="/public/time.svg",
                ),
                cl.Starter(
                    label="新闻搜索",
                    message="2024年八月有什么新闻？",
                    icon="/public/news.svg",
                ),
            ]
        ),
    ]


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
    cl.user_session.set("search_option", None)
    # current_profile = cl.user_session.get("chat_profile")
    # await cl.Message(content="我是商报数智，一个基于" + current_profile + "大模型的问答智能体，请随时向我提问 :)").send()


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
    cl.user_session.set("search_option", None)
    await cl.Message("").send()


# Message handling callback
@cl.on_message
async def on_message(message: cl.Message):
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    message_history = cl.user_session.get("message_history", [])
    search_option = cl.user_session.get("search_option")

    # Get the current chat profile information
    current_profile = cl.user_session.get("chat_profile")

    if search_option is None:
        if current_profile == "llama-3.1-405b & Bing":
            cl.user_session.set("search_option", True)

    message_history.append({"role": "user", "content": message.content})
    cl.user_session.set("message_history", message_history)

    try:
        if cl.user_session.get("search_option"):
            # Bing search API call
            search_results = search(message.content)
            search_prompts = [
                f"来源:\n标题: {result['name']}\n网址: {result['url']}\n内容: {result['snippet']}\n" for result in
                search_results
            ]
            search_content = "Use the following sources to answer the question:\n\n".join(search_prompts) + "\n\nQuestion: " + message.content + "\n\n"

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

# @cl.on_message
# async def on_message(message: cl.Message):
#     memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
#
#     runnable = cl.user_session.get("runnable")  # type: Runnable
#
#     res = cl.Message(content="")
#
#     async for chunk in runnable.astream(
#             {"question": message.content},
#             config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
#     ):
#         await res.stream_token(chunk)
#
#     await res.send()
#
#     memory.chat_memory.add_user_message(message.content)
#     memory.chat_memory.add_ai_message(res.content)
