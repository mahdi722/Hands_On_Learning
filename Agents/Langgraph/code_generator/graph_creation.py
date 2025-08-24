from agent_state import AgentState
from nodes import planner, code_generator
from langgraph.graph import StateGraph, START, END

class Graph:
    def __init__(self):
        self.graph = StateGraph(AgentState)


    def _structure(self):
        self.graph.add_node(node='planner_node', action=planner)
        self.graph.add_node(node='code_generator_node', action=code_generator)

        self.graph.add_edge(start_key=START, end_key='planner_node')
        self.graph.add_edge(start_key='planner_node', end_key='code_generator_node')
        self.graph.add_edge(start_key='code_generator_node', end_key=END)


    def get_graph(self):
        self._structure()
        return self.graph.compile()