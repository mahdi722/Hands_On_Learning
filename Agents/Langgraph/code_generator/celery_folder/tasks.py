from prometheus_client import Counter, Histogram
from celery_folder.celery_app import celery_app
from agent.nodes import code_executor_tool
import time

task_success = Counter("celery_task_success_total", "Successful tasks", ["task"])
task_failure = Counter("celery_task_failure_total", "Failed tasks", ["task"])
task_duration = Histogram("celery_task_duration_seconds", "Task duration", ["task"])

@celery_app.task(name="execute_code")
def execute_code_task(code: str) -> dict:
    start = time.time()
    try:
        result = code_executor_tool(code)
        task_success.labels("execute_code").inc()
        return result
    except Exception as e:
        task_failure.labels("execute_code").inc()
        raise
    finally:
        task_duration.labels("execute_code").observe(time.time() - start)
