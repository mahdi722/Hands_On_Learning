from agent_state import AgentState
from nodes import llm_agent, verifier, starting
from langgraph.graph import StateGraph, START, END

class Graph:
    def __init__(self):
        self.graph = StateGraph(AgentState)

    def _verifier(self, state:AgentState) -> str:
        return state.get('is_ok')
    
    def _structure(self):
        self.graph.add_node(node='starting_node', action=starting)
        self.graph.add_node(node='llm_agent_node', action=llm_agent)
        self.graph.add_node(node='verifier_node', action=verifier)

        self.graph.add_edge(start_key=START, end_key='starting_node')
        self.graph.add_edge(start_key='starting_node', end_key='llm_agent_node')
        self.graph.add_edge(start_key='llm_agent_node', end_key='verifier_node')
        self.graph.add_conditional_edges(source='verifier_node',
                                         path=self._verifier,
                                         path_map={
                                             True:END,
                                             False:'llm_agent_node'
                                         })
    def get_graph(self):
        self._structure()
        return self.graph.compile()