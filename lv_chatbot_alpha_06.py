import streamlit as st
import openai
import pandas as pd
import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from datetime import datetime

def clean_metadata(metadata):
    return {key: (str(value) if isinstance(value, (datetime, pd.Timestamp)) else value) for key, value in metadata.items()}

column_labels = {
    'Edition': 'Edición',
    'No. Monitor (histórico)': 'Monitor Histórico No.',
    'Period (dd/mm)': 'Periodo',
    'Fecha_fin': 'Fecha de Fin',
    'Fecha_inicio': 'Fecha de Inicio',
    'Year': 'Año',
    'TLDR': 'Resumen',
    'Concerns': 'Preocupaciones',
    'text': 'Contenido',
    'Link': 'Enlace'
}

# Configurar la API key de OpenAI (usa variables de entorno en producción)
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Estilos CSS personalizados
st.markdown(
    """
    <style>
    /* Fondo general */
    .stApp {
        background-color: #FCFCF7;
    }

    /* Encabezado */
    .title {
        font-size: 32px;
        font-weight: bold;
        color: #1D252C;
        text-align: center;
        padding: 5px;
        border-bottom: 3px solid #E5FF00;
    }

    /* Mensajes del usuario */
    .message-user {
        background-color: #182828;
        color: white;
        padding: 12px;
        border-radius: 15px;
        margin: 10px 0;
        max-width: 80%;
    }

    /* Mensajes del chatbot */
    .message-bot {
        background-color: #DDD9CE;
        color: #1D252C;
        padding: 12px;
        border-radius: 15px;
        margin: 10px 0;
        max-width: 80%;
    }

    /* Cuadro de entrada de texto */
    .stTextInput > div > div > input {
        background-color: white !important;
        border: 2px solid #182828 !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        padding: 10px !important;
    }

    /* Botón de enviar */
    div.stButton > button {
        background-color: #FF4B4B;
        color: white;
        font-size: 16px;
        border-radius: 8px;
        border: none;
        padding: 8px 16px;
        margin-top: 10px;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #D73838;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Logo de Linterna Verde (si lo quieres agregar)
st.image("logo_horizontal.png", width=150)

# Mensaje de bienvenida
st.markdown('<p class="title">Bienvenido/a al Soros Monitor Chatbot</p>', unsafe_allow_html=True)


# Cargar datos (suponiendo que tienes el archivo localmente)
file_path = "DS_Soros_Monitor.v07.xlsx"
df = pd.read_excel(file_path)

# Crear embeddings y base de datos Chroma
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

documents = [
    Document(
        page_content=" | ".join(
            [f"{column_labels.get(col, col)}: {str(row[col])}" for col in row.index if pd.notna(row[col])]
        ),
        metadata=clean_metadata(row.drop('text').to_dict())
    )
    for _, row in df.iterrows()
]

db = FAISS.from_documents(documents, embeddings) 
retriever = db.as_retriever(search_kwargs={"k": 5})

# Configura el modelo de lenguaje
llm = ChatOpenAI(temperature=0, model="gpt-4-turbo")
qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, return_source_documents=True)

# Interfaz con Streamlit
st.title("Soros Monitor")

# Inicializar el historial de conversación en la sesión
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Mostrar el historial de conversación
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(f'<div class="message-user">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="message-bot">{msg["content"]}</div>', unsafe_allow_html=True)

# Cuadro de entrada para la pregunta
pregunta = st.text_input("Escribe aquí tu pregunta:")

if st.button("Enviar"):
    if pregunta:
        # Guardar la pregunta en el historial
        st.session_state["messages"].append({"role": "user", "content": pregunta})

        # Obtener la respuesta del chatbot
        resultado = qa.invoke({"question": pregunta, "chat_history": st.session_state["messages"]})
        respuesta = resultado["answer"]

        # Guardar la respuesta en el historial
        st.session_state["messages"].append({"role": "assistant", "content": respuesta})

        # Recargar la página para mostrar los mensajes
        st.experimental_rerun()
