import os
from pydantic import BaseModel

class Settings(BaseModel):
    model_name: str = os.getenv("MODEL_NAME", "codellama:latest")
    temperature: float = float(os.getenv("TEMPERATURE", 0.2))
    max_exec_time: int = int(os.getenv("MAX_EXEC_TIME", 5))
    json_logs: bool = os.getenv("JSON_LOGS", "false").lower() == "true"

settings = Settings()
