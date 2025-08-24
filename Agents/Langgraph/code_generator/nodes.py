from agent_state import AgentState
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.tools import Tool
from langchain import hub
from dotenv import load_dotenv
import os
import subprocess
import re
import sys
import importlib

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

PERSISTENT_NAMESPACE = {"__builtins__": __builtins__}


def planner(state: AgentState) -> AgentState:
    prompt = ChatPromptTemplate([
        ("system", "you are a planning agent. user will give you a coding task. Create plans and algorithms for that step by step. No coding just the algorithm in steps. you need to also create test cases and their ideal results so the code generator be able to test with some data"),
        ("user", "{user_query}"),
    ])


    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.2, max_retries=2, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])


    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({'user_query':state['query']})

    state['plan'] = response
    print('_______________________plan________________________')
    print(response)
    return state

def code_generator(state: AgentState) -> AgentState:
    def code_excuter_tool(code_string):
        print("\n EXECUTING:")
        print(code_string)
        print("-" * 40)

        #    (```python ... ```, ``` ... ```)
        fence_re = re.compile(r"^\s*```(?:python)?\s*|\s*```\s*$", re.MULTILINE)
        code_string = fence_re.sub("", code_string).rstrip()

        lines = code_string.split("\n")
        clean_lines = []

        pip_re = re.compile(r"^\s*[!%]?\s*pip\s+install\b", re.IGNORECASE)

        for raw_line in lines:
            s = raw_line.lstrip()

            if pip_re.match(s):
                pkg = re.sub(r"^\s*[!%]?\s*pip\s+install\b", "", raw_line, flags=re.IGNORECASE).strip()
                if not pkg:
                    print("Skipping empty pip install line.")
                    continue
                print(f"Installing {pkg}...")
                try:
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", pkg],
                        capture_output=True, text=True, timeout=300
                    )
                    if result.returncode == 0:
                        print(f"Installed {pkg}")
                    else:
                        print(f"pip failed: {result.stderr}")
                        return f"Failed to install {pkg}: {result.stderr}"
                except Exception as e:
                    return f"Error installing {pkg}: {e}"
            else:
                clean_lines.append(raw_line)


        final_code = "\n".join(clean_lines)
        try:
            exec(final_code, PERSISTENT_NAMESPACE, PERSISTENT_NAMESPACE)
            print("Code executed")
        except Exception as e:
            error = f"Error: {type(e).__name__}: {e}"
            print(f"{error}")
            return error
        
        return "Code executed successfully"
    prompt = ChatPromptTemplate([
        ("system", "you are a code generator agent. read the plan and generate a code string in python for excution. you have access to a tool that excutes python codes. fix the error in case of any error happened. If the code was successful return the code. Remember to return only the code if it was successful"),
        ("user", "plan {plan}"),
        MessagesPlaceholder(variable_name='agent_scratchpad')
    ])


    llm = ChatOpenAI(model='gpt-4o-mini', temperature=float(os.getenv('TEMPERATURE')), max_retries=2, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])


    coder = Tool(
        name='Code_Executer',
        func=code_excuter_tool,
        description="it recieves a string of python code and excutes the code returns the results"
    )
    tools = [coder]
    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
    executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, callbacks=[StreamingStdOutCallbackHandler()], return_intermediate_step=True, max_iteration=5, early_stopping_method="generate"
    )
    response = executor.invoke({"plan": state["plan"]})
    final_code = response.get("output", "") 
    state["code"] = final_code
    print("-------------------------------------------------")
    print(state['code'])
    return state