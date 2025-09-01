from agent_state import AgentState
from nodes import planner, code_generator
from langgraph.graph import StateGraph, START, END

class Graph:
    def __init__(self):
        self.graph = StateGraph(AgentState)

    def _structure(self):
        self.graph.add_node("planner_node", planner)
        self.graph.add_node("code_generator_node", code_generator)
        self.graph.add_edge(START, "planner_node")
        self.graph.add_edge("planner_node", "code_generator_node")
        self.graph.add_edge("code_generator_node", END)

    def get_graph(self):
        self._structure()
        return self.graph.compile()
