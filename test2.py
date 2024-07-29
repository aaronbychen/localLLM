from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain_community.tools.tavily_search.tool import TavilySearchResults
import os
import chainlit as cl
from dotenv import load_dotenv

load_dotenv()

@cl.on_chat_start
async def on_chat_start():
    API_KEY = os.getenv("API_KEY")
    BASE_URL = "https://api.deepbricks.ai/v1/"

    # Initialize Llama 3.1 model
    model = ChatOpenAI(model_name="llama-3.1-405b", api_key=API_KEY, base_url=BASE_URL, streaming=True)

    # Setup the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You're a very knowledgeable historian who provides accurate and eloquent answers to historical questions."),
            ("human", "{question}"),
        ]
    )

    # Setup Tavily search
    search = TavilySearchAPIWrapper()
    tavily_tool = TavilySearchResults(api_wrapper=search)

    # Initialize the agent
    agent_chain = initialize_agent(
        [tavily_tool],
        model,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    # Create runnable with agent
    runnable = prompt | agent_chain | StrOutputParser()
    cl.user_session.set("runnable", runnable)

@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
            {"question": message.content},
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        # Ensure chunk is a string
        if isinstance(chunk, dict):
            chunk = chunk.get("text", "")
        chunk_text = str(chunk)
        await msg.stream_token(chunk_text)

    await msg.send()