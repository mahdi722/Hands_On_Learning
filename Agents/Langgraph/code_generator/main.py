from graph_creation import Graph

flow = Graph()
app = flow.get_graph()
print(app.invoke({'query':'generate a code that merge sorts a list'}))