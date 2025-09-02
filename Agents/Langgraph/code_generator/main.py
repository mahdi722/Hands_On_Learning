from fastapi import FastAPI, HTTPException
from agent.graph_creation import Graph
from agent.agent_state import AgentState
from logger.logger import get_logger
from config import settings
from celery_folder.tasks import execute_code_task
from celery.result import AsyncResult
from prometheus_fastapi_instrumentator import Instrumentator
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from pydantic import BaseModel
trace.set_tracer_provider(TracerProvider())
logger = get_logger("main", json_logs=settings.json_logs)

flow = Graph()
app_graph = flow.get_graph()
app = FastAPI(title="Agent Workflow API", version="1.0.0")

Instrumentator().instrument(app).expose(app)
FastAPIInstrumentor.instrument_app(app)
LoggingInstrumentor().instrument(set_logging_format=True)

class QueryRequest(BaseModel):
    query: str


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

        async_result = execute_code_task.delay(result["code"])

        return {
            "plan": result["plan"],
            "code": result["code"],
            "task_id": async_result.id,
        }

    except Exception as e:
        logger.exception("Unhandled exception in /generate")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    result = AsyncResult(task_id)
    if result.ready():
        return result.result
    return {"status": "pending"}
