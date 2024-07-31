# import os
# from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
# from langchain.agents import initialize_agent, AgentType
# from langchain_openai import ChatOpenAI
# from langchain_community.tools.tavily_search.tool import TavilySearchResults
# from dotenv import load_dotenv
#
# load_dotenv()
# # set up API key
# # os.environ["TAVILY_API_KEY"] = "..."
#
# # set up the agent
# API_KEY = os.getenv("API_KEY")
# BASE_URL = "https://api.deepbricks.ai/v1/"
# llm = ChatOpenAI(model_name="llama-3.1-405b", temperature=0.7, api_key=API_KEY, base_url=BASE_URL)
#
# search = TavilySearchAPIWrapper()
# tavily_tool = TavilySearchResults(api_wrapper=search)
#
# # initialize the agent
# agent_chain = initialize_agent(
#     [tavily_tool],
#     llm,
#     agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True,
# )
#
# # run the agent
# agent_chain.run(
#     "How's today's weather in Beijing?",
# )



import os

from langchain_community.retrievers import TavilySearchAPIRetriever

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = "https://api.deepbricks.ai/v1/"

retriever = TavilySearchAPIRetriever(k=3)

# retriever.invoke("what year was breath of the wild released?")

prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the context provided.

Context: {context}

Question: {question}"""
)
chain = (
        RunnablePassthrough.assign(context=(lambda x: x["question"]) | retriever)
        | prompt
        | ChatOpenAI(model_name="llama-3.1-405b", temperature=0.7, api_key=API_KEY, base_url=BASE_URL)
        | StrOutputParser()
)

chain.invoke({"question": "how many units did breath of the wild sell in 2020"})