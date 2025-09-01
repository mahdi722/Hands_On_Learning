from agent_state import AgentState
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from logger import get_logger
from config import settings


logger = get_logger("nodes", json_logs=settings.json_logs)
PERSISTENT_NAMESPACE = {"__builtins__": __builtins__}

def planner(state: AgentState) -> AgentState:
    """Generate a step-by-step plan for solving the coding task."""
    try:
        prompt = ChatPromptTemplate([
            ("system",
             "You are a planning agent. "
             "The user gives a coding task. "
             "Return a structured step-by-step algorithm and test cases. "
             "Do NOT write Python code."),
            ("user", "{user_query}"),
        ])

        llm = ChatOllama(
            model=settings.model_name,
            temperature=settings.temperature,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        )

        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"user_query": state["query"]})

        state["plan"] = response.strip()
        logger.info("Plan generated successfully")
        return state
    except Exception as e:
        error_msg = f"Planner error: {type(e).__name__}: {e}"
        logger.exception(error_msg)
        state["error"] = error_msg
        return state


def code_executor_tool(code_string: str) -> dict:
    """Execute Python code safely and return stdout/stderr + execution time."""
    import subprocess, sys, time

    start_time = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code_string],
            capture_output=True,
            text=True,
            timeout=settings.max_exec_time,
        )
        exec_time = time.perf_counter() - start_time

        if proc.returncode != 0:
            return {
                "status": "error",
                "stdout": "",
                "stderr": proc.stderr.strip(),
                "execution_time": f"{exec_time:.4f}s"
            }

        return {
            "status": "success",
            "stdout": proc.stdout.strip() or "Execution succeeded",
            "stderr": "",
            "execution_time": f"{exec_time:.4f}s"
        }

    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "stdout": "",
            "stderr": f"TimeoutError: exceeded {settings.max_exec_time}s",
            "execution_time": f"{settings.max_exec_time:.4f}s"
        }

    except Exception as e:
        exec_time = time.perf_counter() - start_time
        return {
            "status": "error",
            "stdout": "",
            "stderr": str(e),
            "execution_time": f"{exec_time:.4f}s"
        }


def code_generator(state: AgentState) -> AgentState:
    """Use ChatGPT with tool calling to generate + test code."""
    try:
        prompt = ChatPromptTemplate([
            ("system",
             "You are a coding agent. "
             "The user will give you a plan that includes algorithm steps and test cases. "
             "Generate Python code. "
             "Use the CodeExecutor tool to run the code against the test cases. "
             "If tests fail, fix the code and try again. "
             "Return ONLY the final passing Python code."),
            ("user", "{plan}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        llm = ChatOpenAI(
            model="gpt-4o-mini",   
            temperature=settings.temperature,
        )

        tools = [
            Tool(
                name="CodeExecutor",
                func=code_executor_tool,
                description="Executes Python code and returns stdout/stderr",
            )
        ]

        agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)

        executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            return_intermediate_steps=True,
            max_iterations=5,
            early_stopping_method="generate",
            handle_parsing_errors=True,  
        )

        logger.info("Starting code generation with ChatGPT agent...")
        response = executor.invoke({"plan": state["plan"]})

        state["code"] = response.get("output", "")
        state["result"] = response.get("intermediate_steps", "")
        state["execution_time"] = None
        state["error"] = ""
        return state

    except Exception as e:
        error_msg = f"Code generator error: {type(e).__name__}: {e}"
        logger.exception(error_msg)
        state["error"] = error_msg
        return state