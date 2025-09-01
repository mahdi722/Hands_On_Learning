import os
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

class Settings(BaseModel):
    model_name: str = os.getenv("MODEL_NAME", "codellama:latest")
    temperature: float = float(os.getenv("TEMPERATURE", 0.2))
    max_exec_time: int = int(os.getenv("MAX_EXEC_TIME", 5))
    json_logs: bool = os.getenv("JSON_LOGS", "false").lower() == "true"
    openai_api: str = os.getenv("OPENAI_API_KEY")

settings = Settings()
os.environ['OPENAI_API_KEY'] = settings.openai_api
