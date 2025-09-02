import pytest
import subprocess
from unittest.mock import patch, MagicMock

from agent.nodes import planner, code_executor_tool, code_generator, AgentState


@pytest.fixture
def dummy_state() -> AgentState:
    return {
        "query": "Write bubble sort",
        "plan": "",
        "error": "",
        "code": "",
        "result": "",
    }


@patch("agent.nodes.ChatOllama")
@patch("agent.nodes.StrOutputParser")
def test_planner_success(mock_parser, mock_chatollama, dummy_state):
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = "Step 1: Do this"
    state = planner(dummy_state, chain_factory=lambda: fake_chain)
    assert "Step" in state["plan"]



@patch("agent.nodes.ChatOllama", side_effect=Exception("Ollama not available"))
def test_planner_failure(mock_chatollama, dummy_state):
    state = planner(dummy_state)
    assert "error" in state
    assert "Planner error" in state["error"]

@patch("subprocess.run")
def test_code_executor_tool_success(mock_run):
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = "Hello"
    mock_proc.stderr = ""
    mock_run.return_value = mock_proc

    result = code_executor_tool('print("Hello")')

    assert result["status"] == "success"
    assert "Hello" in result["stdout"]


@patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="python", timeout=1))
def test_code_executor_tool_timeout(mock_run):
    result = code_executor_tool("while True: pass")
    assert result["status"] == "timeout"
    assert "TimeoutError" in result["stderr"]


@patch("agent.nodes.AgentExecutor.from_agent_and_tools")
@patch("agent.nodes.create_openai_functions_agent")
@patch("agent.nodes.ChatOpenAI")
def test_code_generator_success(mock_chatopenai, mock_create_agent, mock_from_agent, dummy_state):
    dummy_state["plan"] = "Step 1: implement add function"

    mock_executor = MagicMock()
    mock_executor.invoke.return_value = {
        "output": "def add(a,b): return a+b",
        "intermediate_steps": ["test passed"],
    }
    mock_from_agent.return_value = mock_executor

    state = code_generator(dummy_state)

    assert "code" in state
    assert "def add" in state["code"]
    assert state["error"] == ""


@patch("agent.nodes.AgentExecutor.from_agent_and_tools", side_effect=Exception("Bad agent"))
@patch("agent.nodes.create_openai_functions_agent")
@patch("agent.nodes.ChatOpenAI")
def test_code_generator_failure(mock_chatopenai, mock_create_agent, mock_from_agent, dummy_state):
    dummy_state["plan"] = "invalid plan"

    state = code_generator(dummy_state)

    assert "error" in state
    assert "Code generator error" in state["error"]
