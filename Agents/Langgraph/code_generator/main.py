from fastapi import FastAPI, HTTPException
from graph_creation import Graph
from agent_state import AgentState
from logger import get_logger
from config import settings
from request_models import QueryRequest
logger = get_logger("main", json_logs=settings.json_logs)

flow = Graph()
app_graph = flow.get_graph()
app = FastAPI(title="Agent Workflow API", version="1.0.0")

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/generate")
async def invoke_agent(request: QueryRequest):
    try:
        state: AgentState = {
            "query": request.query,
            "error": "",
            "code": "",
            "plan": "",
            "result": "",
        }
        result = app_graph.invoke(state)
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
        return {
            "plan": result.get("plan"),
            "code": result.get("code"),
            "execution_result": result.get("result"),
            "execution_time": result.get("execution_time"),
        }

    except Exception as e:
        logger.exception("Unhandled exception in /generate")
        raise HTTPException(status_code=500, detail=str(e))
