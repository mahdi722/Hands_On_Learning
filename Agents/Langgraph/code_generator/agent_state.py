from typing import TypedDict


class AgentState(TypedDict):
    query: str
    error: str
    code: str
    plan: str