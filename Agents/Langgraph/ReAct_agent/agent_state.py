from typing import TypedDict, List

class AgentState(TypedDict):
    message: str
    is_ok: bool
    history: List
    answer : str
    counter: int