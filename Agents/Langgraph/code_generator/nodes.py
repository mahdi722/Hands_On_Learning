from agent_state import AgentState
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from logger import get_logger
from dotenv import load_dotenv
load_dotenv()
from config import settings

import subprocess, re, ast, time

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


def safe_execute(code_string: str) -> str:
    """Run code in a sandboxed subprocess with timeout."""
    try:
        proc = subprocess.run(
            ["python3", "-c", code_string],
            capture_output=True,
            text=True,
            timeout=settings.max_exec_time,
        )
        if proc.returncode != 0:
            return f"RuntimeError: {proc.stderr.strip()}"
        return proc.stdout.strip() if proc.stdout.strip() else "No output"
    except subprocess.TimeoutExpired:
        return f"TimeoutError: exceeded {settings.max_exec_time}s"



def code_generator(state: AgentState) -> AgentState:
    """Generate and safely execute Python code from the plan."""
    try:
        prompt = ChatPromptTemplate([
            ("system",
             "You are a Python code generator. "
             "Return ONLY valid Python code. "
             "Wrap it inside triple backticks with 'python'. "
             "Do NOT add explanations, comments outside code, "
             "or tags like [/PYTHON]."),
            ("user", "{plan}"),
        ])

        llm = ChatOllama(
            model=settings.model_name,
            temperature=settings.temperature,
            streaming=True,
        )

        chain = prompt | llm | StrOutputParser()
        raw_output = chain.invoke({"plan": state["plan"]})

        match = re.search(r"```(?:python)?\s*([\s\S]*?)```", raw_output)
        code_string = match.group(1).strip() if match else raw_output.strip()

        code_string = re.sub(r"\[/?PYTHON\]", "", code_string)

        logger.info(f"Generated code:\n{code_string}")

        try:
            ast.parse(code_string)
        except SyntaxError as e:
            state["error"] = f"Invalid Python generated: {e}"
            logger.error(state["error"])
            return state

        start_time = time.perf_counter()
        result = safe_execute(code_string)
        exec_time = time.perf_counter() - start_time

        logger.info(f"Execution finished in {exec_time:.4f} seconds")

        state["code"] = code_string
        state["result"] = result
        state["execution_time"] = f"{exec_time:.4f}s"
        state["error"] = "" if "Error" not in result else result
        return state

    except Exception as e:
        error_msg = f"Code generator error: {type(e).__name__}: {e}"
        logger.exception(error_msg)
        state["error"] = error_msg
        return state

