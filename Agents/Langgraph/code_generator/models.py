from pydantic import BaseModel, ValidationError
from typing import List, Any, Dict

class QueryRequest(BaseModel):
    query: str


class TestCase(BaseModel):
    input: Any
    output: Any

