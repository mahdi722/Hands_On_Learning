from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from agent_state import AgentState
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

def starting_node(state: AgentState) -> AgentState:
    state['memory'] = []
    return state
def processor(state: AgentState) -> AgentState:

    while True:
        msg = input('TEXT : ')
        if msg == 'exit':
            return state
        state['message'] = msg
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are a conversational Agent with memory. \n Memory : {str(state['memory'])}. Listen to user and answer to their question. Carefull about saftey alignment"),
            ("user", f"User_message : {state['message']}")
        ])

        state['memory'].append('user  :  ' + state['message'])
        llm = ChatGoogleGenerativeAI(model=os.getenv('MODEL'), temperature=os.getenv('TEMPERATURE'))


        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({})
        print(response)
        state['memory'].append('Agent  :  ' + response)

    

