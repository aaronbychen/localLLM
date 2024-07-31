from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate

from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun

from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import re
import langchain
import os
import chainlit as cl


from dotenv import load_dotenv

load_dotenv()

# os.environ["OPENAI_API_KEY"] = "{Your_API_Key}"

'''
template = """
As a local travel tour professional, you will answer your questions to the best of your ability. You have access to the following tools:

{tools}

Use the following format:

Question: The question you have to answer
Thought: Your thought process in approaching the question
Action: Choose one of the available tools in [{tool_names}] for your action
Action Input: Provide the input required for the chosen tool
Observation: Describe the result obtained from the action
...(Repeat several times of the Thought/Action/Action Input/Observation as needed)
Thought: Now I have the final answer!
Final Answer: Provide your final answer from the perspective of an experienced local travel tour professional

Let's get started!
Question: {input}
{agent_scratchpad}"""
'''
template = """Answer the following questions as best you can, but speaking as passionate travel expert. You have access to the following tools:

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
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)


class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


def search_online(input_text):
    search = DuckDuckGoSearchRun().run(f"site:tripadvisor.com things to do{input_text}")
    return search


def search_hotel(input_text):
    search = DuckDuckGoSearchRun().run(f"site:booking.com {input_text}")
    return search


def search_flight(input_text):
    search = DuckDuckGoSearchRun().run(f"site:skyscanner.com {input_text}")
    return search


def search_general(input_text):
    search = DuckDuckGoSearchRun().run(f"{input_text}")
    return search


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
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=["input", "intermediate_steps"]
    )

    output_parser = CustomOutputParser()
    API_KEY = os.getenv("API_KEY")
    BASE_URL = "https://api.deepbricks.ai/v1/"
    llm = ChatOpenAI(model_name="llama-3.1-405b", api_key=API_KEY, base_url=BASE_URL, temperature=0)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    tool_names = [tool.name for tool in tools]

    agent = LLMSingleActionAgent(llm_chain=llm_chain, output_parser=output_parser, stop=["\nObservation:"],
                                 allowed_tools=tool_names)
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent,
                                                        tools=tools,
                                                        verbose=True)
    cl.user_session.set("agent", agent_executor)


@cl.on_message
async def main(message):
    agent = cl.user_session.get("agent")  # type: AgentExecutor
    cb = cl.LangchainCallbackHandler(stream_final_answer=True)

    await cl.make_async(agent.run)(message, callbacks=[cb])
