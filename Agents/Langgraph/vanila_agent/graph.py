from langgraph.graph import StateGraph, START, END
from nodes import starting_node, processor
from agent_state import AgentState

class work_flow():
    def __init__(self):
        self.graph = StateGraph(AgentState)

    def _define_nodes(self):
        self.graph.add_node(node='Starting_Node', action=starting_node)
        self.graph.add_node(node='Chat_Node', action=processor)

    def _define_edges(self):
        self.graph.add_edge(start_key=START, end_key='Starting_Node')
        self.graph.add_edge(start_key='Starting_Node', end_key='Chat_Node')
        self.graph.add_edge(start_key='Chat_Node', end_key=END)

    def get_graph(self):
        self._define_nodes()
        self._define_edges()
        return self.graph.compile()