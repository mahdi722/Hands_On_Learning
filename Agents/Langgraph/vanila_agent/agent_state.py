from typing import TypedDict, List, Annotated


class AgentState(TypedDict):
    message: str
    memory: Annotated[List, "memory of interactions"]

