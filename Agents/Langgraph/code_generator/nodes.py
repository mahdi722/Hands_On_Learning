from agent_state import AgentState
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import Tool
from langchain import hub
from dotenv import load_dotenv
import os

load_dotenv()

os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')



def planner(state: AgentState) -> AgentState:
    prompt = ChatPromptTemplate([
        ("system", "you are a planning agent. user will give you a coding task. Create plans and algorithms for that step by step. No coding just the algorithm in steps. you need to also create test cases and their ideal results so the code generator be able to test with some data"),
        ("user", "{user_query}"),
    ])


    llm = ChatGoogleGenerativeAI(model=os.getenv('MODEL'), temperature=0.2, max_retries=2)


    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({'user_query':state['query']})

    state['plan'] = response
    print(state)
    return state

def code_generator(state: AgentState) -> AgentState:
    def code_excuter_tool(code_string):
        try:
            result = exec(code_string)
            return result
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"
        
    prompt = ChatPromptTemplate([
        ("system", "you are a code generator agent. read the plan and generate a code string in python for excution. you have access to a tool that excutes python codes. fix the error in case of any error happened. If the code was successful return the code"),
        ("user", "plan {plan}"),
        MessagesPlaceholder(variable_name='agent_scratchpad')
    ])


    llm = ChatGoogleGenerativeAI(model=os.getenv('MODEL'), temperature=float(os.getenv('TEMPERATURE')), max_retries=2)


    coder = Tool(
        name='Code_Executer',
        func=code_excuter_tool,
        description="it recieves a string of python code and excutes the code returns the results"
    )
    tools = [coder]
    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
    executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools
    )
    response = executor.invoke({"plan": state["plan"]})
    # AgentExecutor returns a dict; usually response["output"] has the final string
    final_code = response.get("output", "")
    state["code"] = final_code
    print(state)
    return state