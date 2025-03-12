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
        font-size: 52px;
        font-weight: bold;
        color: #1D252C;
        text-align: center;
        padding: 5px;
        border-bottom: 3px solid #E5FF00;
    }
    /* Contenedor del banner */
    .banner {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        background-color: #FCFCF7;
        padding: 10px 20px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        z-index: 1000;
        text-align: center;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
    }

    /* Imagen del logo */
    .banner .logo {
        width: 150px; /* Ajusta el tamaño del logo */
        height: auto;
        margin-bottom: 5px;
    }

    /* Subtítulo */
    .banner p {
        font-size: 18px;
        font-weight: bold;
        margin: 5px 0;
    }

    /* Línea inferior */
    .banner .underline {
        width: 100%;
        height: 3px;
        background-color: #E5FF00;
        margin-top: 5px;
    }

    /* Ajustar el espacio del contenido para que no se solape con el banner */
    .content {
        margin-top: 100px; /* Ajusta este valor según la altura del banner */
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

    /* Contenedor del input y el botón */
    .chat-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        width: 80%;
        margin: 10px auto;
    }

    /* Cuadro de entrada de texto */
    .chat-input {
        flex-grow: 1;
        background-color: white !important;
        border: 2px solid #182828 !important;
        color: #1D252C !important;
        border-radius: 15px !important;
        padding: 12px !important;
        font-size: 16px;
        outline: none;
        width: 100%;
    }

    /* Botón de enviar */
    .send-button {
        background: none;
        border: 2px solid #182828;
        border-radius: 8px;
        cursor: pointer;
        margin-left: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 48px;
        height: 48px;
        background-color: #182828;
    }
    
    .send-button img {
        width: 40px;
        height: 40px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="banner">
        <img src="https://github.com/laurapaqu/Soros-Monitor-streamlit/blob/main/logo_horizontal.png?raw=true" class="logo">
        <p>Bienvenido/a al Soros Monitor Chatbot</p>
        <div class="underline"></div>
    </div>
    <div class="content">
    """,
    unsafe_allow_html=True
)

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


# Inicializar el historial de conversación en la sesión
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Mostrar el historial de conversación
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(f'<div class="message-user">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="message-bot">{msg["content"]}</div>', unsafe_allow_html=True)

# Espacio para la entrada de texto y el botón de imagen
st.markdown(
    """
    <div class="chat-container">
        <input type="text" id="user_input" class="chat-input" placeholder="Escribe aquí tu pregunta...">
        <button class="send-button" onclick="sendMessage()">
            <img src='https://github.com/laurapaqu/Soros-Monitor-streamlit/blob/main/send.png?raw=true' alt="Enviar">
        </button>
    </div>
    <script>
        function sendMessage() {
            let inputField = document.getElementById("user_input");
            let userMessage = inputField.value;
            if (userMessage.trim() !== "") {
                inputField.value = "";
                window.location.href = "/?query=" + encodeURIComponent(userMessage);
            }
        }
    </script>
    """,
    unsafe_allow_html=True
)


# Capturar la pregunta desde la URL si el usuario presionó el botón
query_param = st.query_params.get("query", "")
if query_param:
    pregunta = query_param
    st.session_state["messages"].append({"role": "user", "content": pregunta})

    # Obtener la respuesta del chatbot
    resultado = qa.invoke({"question": pregunta, "chat_history": st.session_state["messages"]})
    respuesta = resultado["answer"]

    # Guardar la respuesta en el historial
    st.session_state["messages"].append({"role": "assistant", "content": respuesta})

    # Recargar la página para mostrar los mensajes
    st.experimental_rerun()