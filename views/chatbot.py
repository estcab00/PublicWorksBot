import os
import json
import streamlit as st
import re
import openai
from dotenv import load_dotenv

# Imports para LangChain, LangGraph y herramientas de ML
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from typing import Literal

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState, END
from langgraph.types import Command

# Imports para TF-IDF y similitud coseno
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from tools import search_relevant_chunks_tool_new, find_relevant_chunks_tfidf_2step


# Streamlit configuration
with st.sidebar:
    st.markdown(
        """
        <h1 style="color: white;">Interactive Chatbot</h1>
        <h2 style="color: white;">About this chatbot</h2>
        <p style="color: white;">
        This is a chatbot that allows you to ask questions regarding different public work contracts in Peru.
        </p>
        """,
        unsafe_allow_html=True
    )

# Cargar variables de entorno desde el archivo .env
load_dotenv()
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Verificar que la clave de OpenAI esté configurada
if not openai_api_key:
    st.error("La clave de OpenAI no está configurada. Por favor, revisa tus variables de entorno.")
    st.stop()

def make_system_prompt(suffix: str) -> str:
    return (
        "You are a helpful AI assistant, collaborating with other assistants."
        " Use the provided tools to progress towards answering the question regarding different public projects in Peru."
        " If you are unable to fully answer, that's OK, another assistant with different tools "
        " will help where you left off. Execute what you can to make progress."
        " If you or any of the other assistants have the final answer or deliverable,"
        " prefix your response with FINAL ANSWER so the team knows to stop."
        f"\n{suffix}"
    )

# # Inicializar el estado de la sesión para retener los mensajes
# if "messages" not in st.session_state:
#     st.session_state["messages"] = []

# prompt = st.text_input("Your inquiry:", "")


llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)

def get_next_node(last_message: BaseMessage, goto: str):
    if "FINAL ANSWER" in last_message.content:
        # Any agent decided the work is done
        return END
    return goto


# Research agent and node
research_agent = create_react_agent(
    llm,
    tools=[search_relevant_chunks_tool_new],
    state_modifier=make_system_prompt(
        "You can only find the most relevant JSON files and their most relevant chunks of information with the tool provided. You are working with an answer generator colleague."
    ),
)


def research_node(
    state: MessagesState,
) -> Command[Literal["chart_generator", END]]:
    result = research_agent.invoke(state)
    goto = get_next_node(result["messages"][-1], "chart_generator")
    # wrap in a human message, as not all providers allow
    # AI message at the last position of the input messages list
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="researcher"
    )
    return Command(
        update={
            # share internal message history of research agent with other agents
            "messages": result["messages"],
        },
        goto=goto,
    )

# Chart generator agent and node
chart_agent = create_react_agent(
    llm,
    tools=[],
    state_modifier=make_system_prompt(
        "You can only give the FINAL answer based on the information provided by your colleague. You give a concise and direct FINAL answer showing that you dominate the topic. You are working with a researcher colleague that will find the relevant information for you to answer the question."
    ),
)

def chart_node(state: MessagesState) -> Command[Literal["researcher", END]]:
    result = chart_agent.invoke(state)
    goto = get_next_node(result["messages"][-1], "researcher")
    # wrap in a human message, as not all providers allow
    # AI message at the last position of the input messages list
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="chart_generator"
    )
    return Command(
        update={
            # share internal message history of chart agent with other agents
            "messages": result["messages"],
        },
        goto=goto,
    )


from langgraph.graph import StateGraph, START

graph = StateGraph(MessagesState)
graph.add_node("researcher", research_node)
graph.add_node("chart_generator", chart_node)

graph.add_edge(START, "researcher")
app = graph.compile()


# --------------------------
# 5. Función para ejecutar el workflow con la pregunta y directorio ingresados
# --------------------------
def run_workflow(question: str, json_directory: str) -> str:
    # Se crea el mensaje de entrada para el workflow
    input_message = HumanMessage(
        content=f"Tengo la siguiente pregunta: {question}. "
                f"En caso sea necesario, utiliza los datos de {json_directory} para responder."
    )
    config = {"recursion_limit": 150, "configurable": {"thread_id": "1"}}
    
    # Usamos el objeto compilado 'app' en lugar de 'graph'
    output = app.invoke({"messages": [input_message]}, config=config)
    
    # Se recorre la lista de mensajes para obtener la respuesta final (se asume que el último mensaje contiene FINAL ANSWER)
    final_answer = None
    for msg in output["messages"]:
        if "FINAL ANSWER" in msg.content:
            final_answer = msg.content
            break
    if not final_answer:
        final_answer = output["messages"][-1].content  # En caso no se haya detectado "FINAL ANSWER"
    return final_answer

# --------------------------
# 6. Interfaz Streamlit
# --------------------------
def main():
    st.title("Chatbot de Proyectos Públicos en Perú (LangGraph)")
    
    st.markdown(
        """
        Esta aplicación utiliza un workflow colaborativo basado en LangGraph para responder preguntas
        sobre proyectos públicos en Perú. Ingresa tu pregunta y la ruta del directorio donde se encuentran
        los archivos JSON con la información del proyecto.
        """
    )

st.markdown(
    """
    <style>
    /* Estiliza los cuadros de entrada (text_input) y las áreas de texto (text_area) */
    input[type="text"], textarea {
         background-color: #25306c !important;  /* Fondo azul oscuro para widgets interactivos */
         color: #ffffff !important;             /* Texto en blanco */
         border: 1px solid #2a5298 !important;   /* Borde en azul medio, para dar consistencia */
    }
    
    /* Opcional: Cambiar el color del placeholder a blanco (con opacidad) */
    ::placeholder {
         color: #ffffff !important;
         opacity: 0.7 !important;
    }
    </style>

    /* Estilo para las etiquetas de los text_input */
    div.stTextInput label, div[data-testid="stTextInput"] label {
         color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)    
    
# Entradas para la pregunta y el directorio JSON
question = st.text_input("Ingresa tu pregunta:", "")
json_directory = st.text_input("Ingresa la ruta del directorio JSON:", "")

if st.button("Enviar"):
    if question.strip() == "" or json_directory.strip() == "":
        st.error("Por favor ingresa tanto la pregunta como la ruta del directorio JSON.")
    else:
        with st.spinner("Generando respuesta..."):
            answer = run_workflow(question, json_directory)
        st.markdown("### Respuesta")
        st.text_area("Respuesta final", value=answer, height=200)

if __name__ == "__main__":
    main()


