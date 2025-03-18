import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_community.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

# Cargar API Key de OpenAI
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Estilos CSS personalizados
st.markdown(
    """
    <style>
    .stApp {
        background-color: #FCFCF7;
    }
    
    /* Título */
    .title {
        font-size: 32px;
        font-weight: bold;
        color: #1D252C;
        text-align: center;
        padding: 5px;
        border-bottom: 3px solid #E5FF00;
    }

    /* Contenedor del chat */
    .chat-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-top: 20px;
        gap: 10px;
    }

    /* Cuadro de entrada */
    .chat-input {
        flex-grow: 1;
        border: 2px solid #E5FF00;
        border-radius: 8px;
        padding: 10px;
        font-size: 16px;
        width: 80%;
    }

    /* Botón de enviar con imagen */
    .send-button {
        background-color: #182828;
        border-radius: 8px;
        padding: 8px;
        cursor: pointer;
        border: none;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .send-button img {
        width: 30px; /* Ajusta el tamaño de la imagen */
        height: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Mensaje de bienvenida
st.markdown('<h1 class="title">Bienvenido/a al Soros Monitor Chatbot</h1>', unsafe_allow_html=True)

# Configurar el modelo de lenguaje
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=openai_api_key)

# Cargar la base de datos FAISS con embeddings
embeddings = OpenAIEmbeddings()
db = FAISS.load_local("faiss_index", embeddings)
retriever = db.as_retriever()
qa = ConversationalRetrievalChain.from_llm(llm, retriever)

# Inicializar el historial de conversación en la sesión
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Mostrar el historial de conversación
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(f'<div class="message-user">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="message-bot">{msg["content"]}</div>', unsafe_allow_html=True)

# 📌 **Capturar la entrada del usuario**
col1, col2 = st.columns([4, 1])

with col1:
    pregunta = st.text_input("Escribe aquí tu pregunta:", key="input_text", label_visibility="collapsed")

with col2:
    if st.button("Enviar"):
        if pregunta.strip():
            # Guardar la pregunta en la sesión
            st.session_state["messages"].append({"role": "user", "content": pregunta})

            # Obtener la respuesta del chatbot
            resultado = qa.invoke({"question": pregunta, "chat_history": st.session_state["messages"]})
            respuesta = resultado["answer"]

            # Guardar la respuesta en la sesión
            st.session_state["messages"].append({"role": "assistant", "content": respuesta})

            # Recargar la página para mostrar los mensajes
            st.experimental_rerun()
