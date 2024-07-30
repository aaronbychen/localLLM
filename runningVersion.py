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
import os
from dotenv import load_dotenv

load_dotenv()

# DB_PWD = os.getenv("DB_PWD")

# 数据库配置
DB_CONFIG = {
    'user': 'root',
    'password': os.getenv("DB_PWD"),
    'host': '127.0.0.1',
    'database': 'user_db',
}


# 获取数据库连接
def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)


# 读取用户信息
def read_user(username):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT password_hash FROM users WHERE username = %s", (username,))
    user = cursor.fetchone()
    conn.close()
    return user


# 写入用户信息
def write_user(username, password_hash):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s)", (username, password_hash))
    conn.commit()
    conn.close()


# 配置API客户端
API_KEY = os.getenv("API_KEY")
BASE_URL = "https://api.deepbricks.ai/v1/"
client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
settings = {
    "model": "llama-3.1-405b",
    "temperature": 0.7,
    "max_tokens": 4095,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}


# 设置可运行模型
def setup_runnable():
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    model = ChatOpenAI(streaming=True)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. You reply every dialogue in Chinese."),
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


# 用户认证回调
@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    user = read_user(username)
    if user and bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
        return cl.User(identifier=username, metadata={"role": "user", "provider": "credentials"})
    elif not user:
        # 自动注册用户
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        write_user(username, password_hash)
        return cl.User(identifier=username, metadata={"role": "user", "provider": "credentials"})
    else:
        return None


# 聊天开始回调
@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))
    setup_runnable()
    await cl.Message(content="我是基于" + settings.get("model") + "的API接口的聊天机器人，请随时向我提问 :)").send()


# 聊天恢复回调
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


# 消息处理回调
@cl.on_message
async def on_message(message: cl.Message):
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    message_history = cl.user_session.get("message_history", [])

    message_history.append({"role": "user", "content": message.content})
    cl.user_session.set("message_history", message_history)

    try:
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

    # 更新内存中的聊天历史
    memory.chat_memory.add_user_message(message.content)
    memory.chat_memory.add_ai_message(msg.content)

    # 保存会话状态
    cl.user_session.set("memory", memory)
    cl.user_session.set("message_history", message_history)
