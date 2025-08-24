from graph_creation import Graph

flow = Graph()
app = flow.get_graph()

print(app.invoke({'message':'what is the temperature of Rome'}))