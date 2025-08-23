# graph.py
from langgraph.graph import StateGraph, START, END
from nodes import starting_node, processor, AgentState

class WorkFlow:
    def __init__(self):
        self.graph = StateGraph(AgentState)

    def _define(self):
        self.graph.add_node("Starting_Node", starting_node)
        self.graph.add_node("Chat_Node", processor)
        self.graph.add_edge(START, "Starting_Node")
        self.graph.add_edge("Starting_Node", "Chat_Node")
        self.graph.add_edge("Chat_Node", END)

    def get_graph(self):
        self._define()
        return self.graph.compile()
