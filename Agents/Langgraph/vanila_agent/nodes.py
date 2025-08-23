# nodes.py
from typing import TypedDict, List, Annotated
import operator, os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

class AgentState(TypedDict):
    message: str
    memory: Annotated[List[str], operator.add]
    response: str

def starting_node(state: AgentState) -> AgentState:
    return state

def processor(state: AgentState) -> AgentState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a conversational agent with memory.\nMemory: {memory}\nBe helpful and safe."),
        ("user", "User message: {message}")
    ])

    mem = list(state.get("memory", []))
    mem.append(f"user: {state['message']}")

    model = os.getenv("MODEL") or "gemini-2.5-flash"
    temperature = float(os.getenv("TEMPERATURE") or 0.2)
    llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)

    chain = prompt | llm | StrOutputParser()
    reply = chain.invoke({"memory": str(mem), "message": state["message"]})

    mem.append(f"agent: {reply}")
    state["memory"] = mem
    state["response"] = reply
    return state
