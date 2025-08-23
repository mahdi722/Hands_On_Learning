from graph import work_flow

flow = work_flow()
app = flow.get_graph()
app.invoke({})