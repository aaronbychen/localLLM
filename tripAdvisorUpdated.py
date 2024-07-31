from langchain.agents import Tool, AgentExecutor, create_react_agent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import re
import os
import chainlit as cl
from dotenv import load_dotenv

load_dotenv()

template = """Answer the following questions as best you can, but speaking as a passionate travel expert. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to answer as a passionate and informative travel expert when giving your final answer.

Question: {input}
{agent_scratchpad}"""


class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)


class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


def search_online(input_text):
    return DuckDuckGoSearchRun().run(f"site:tripadvisor.com things to do {input_text}")


def search_hotel(input_text):
    return DuckDuckGoSearchRun().run(f"site:booking.com {input_text}")


def search_flight(input_text):
    return DuckDuckGoSearchRun().run(f"site:skyscanner.com {input_text}")


def search_general(input_text):
    return DuckDuckGoSearchRun().run(f"{input_text}")


@cl.on_chat_start
def start():
    tools = [
        Tool(
            name="Search general",
            func=search_general,
            description="useful for when you need to answer general travel questions"
        ),
        Tool(
            name="Search tripadvisor",
            func=search_online,
            description="useful for when you need to answer trip plan questions"
        ),
        Tool(
            name="Search booking",
            func=search_hotel,
            description="useful for when you need to answer hotel questions"
        ),
        Tool(
            name="Search flight",
            func=search_flight,
            description="useful for when you need to answer flight questions"
        )
    ]

    prompt = CustomPromptTemplate(
        template=template,
        tools=tools,
        input_variables=["input", "intermediate_steps", "tools", "tool_names", "agent_scratchpad"]
    )

    output_parser = CustomOutputParser()
    API_KEY = os.getenv("API_KEY")
    BASE_URL = "https://api.deepbricks.ai/v1/"
    llm = ChatOpenAI(model_name="llama-3.1-405b", api_key=API_KEY, base_url=BASE_URL, temperature=0)

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
        output_parser=output_parser
    )

    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
    cl.user_session.set("agent", agent_executor)


@cl.on_message
async def main(message):
    agent = cl.user_session.get("agent")  # type: AgentExecutor
    cb = cl.LangchainCallbackHandler(stream_final_answer=True)
    inputs = {"input": message}
    await cl.make_async(agent.invoke)(inputs, callbacks=[cb])
