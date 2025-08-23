from typing import List, Optional, Dict
from fastapi import FastAPI, Header
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from graph import WorkFlow

app = FastAPI()
compiled_graph = WorkFlow().get_graph()
SESSION_STORE: Dict[str, List[str]] = {}

class AgentRequest(BaseModel):
    message: str

class AgentResponse(BaseModel):
    response: str
    memory: List[str]

@app.post("/agent", response_model=AgentResponse)
async def agent(body: AgentRequest, session_id: str = Header(default="default")):
    mem = SESSION_STORE.get(session_id, [])
    result = await run_in_threadpool(compiled_graph.invoke, {"message": body.message, "memory": mem})
    SESSION_STORE[session_id] = result["memory"]
    return AgentResponse(response=result["response"], memory=result["memory"])
