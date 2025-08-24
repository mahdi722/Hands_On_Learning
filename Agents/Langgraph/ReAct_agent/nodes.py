from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_tavily import TavilySearch
from agent_state import AgentState
from langchain.agents import AgentExecutor, create_openai_functions_agent
from dotenv import load_dotenv
import os

load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")


def starting(state: AgentState) -> AgentState:
    state['history'] = []
    state['counter'] = 0
    return state
def llm_agent(state: AgentState) -> AgentState:
    tavily = TavilySearch(
    max_results=5,
    topic="general",
)
    tools = [tavily]
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You can call tools when needed (Tavily search). "
         "Follow ReAct: think, use tools if necessary, then give a final concise answer."),
        ("human", "History:\n{history}\n\nQuestion: {query}"),
        # Required placeholder for intermediate tool thoughts/steps
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    llm = ChatGoogleGenerativeAI(model=os.getenv('MODEL'),temperature=os.getenv('TEMPERATURE'))


    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)

    # Wrap with an executor that runs the tool loop
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=4,
        handle_parsing_errors=True,
    )

    response = executor.invoke({'query':state['message'], 'history':state['history']})
    state['answer'] = response
    state['history'].append(response)
    print(state)
    return state

def verifier(state: AgentState) -> AgentState:
    prompt = ChatPromptTemplate([
        ('system','you are a verifier. check if the answer of the previous agent make sense the answer only must be a word. good if the answer os ok, bad if the answer has any problem. Do not generate anything else. the output can only be good or bad'),
        ('user','output \n : {output}')
    ])

    llm = ChatGoogleGenerativeAI(model=os.getenv('MODEL'),temperature=os.getenv('TEMPERATURE'))

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({'output':state['answer']})

    if response == 'good' or state['counter'] >= 1:
        print(state)
        state['is_ok'] = True
    elif response == 'bad':
        print(state)
        state['counter'] += 1
        state['is_ok'] = False
    else:
        raise "the out put of verifier is not ligible"
    print(state)
    return state