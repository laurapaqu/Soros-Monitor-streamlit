import streamlit as st
import openai  # Si usas OpenAI, asegúrate de configurar tu API Key
import pandas as pd
import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import Chroma
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

db = Chroma.from_documents(documents, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 5})
st.write(f"📂 Cantidad de documentos indexados en ChromaDB: {len(documents)}")

# Configura el modelo de lenguaje
llm = ChatOpenAI(temperature=0, model="gpt-4-turbo")
qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, return_source_documents=True)

# Interfaz con Streamlit
st.title("Chatbot con Streamlit")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Mostrar el historial de mensajes
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Entrada del usuario
if prompt := st.chat_input("Haz una pregunta..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Obtener respuesta del chatbot
    chat_history_tuples = [(msg["role"], msg["content"]) for msg in st.session_state["messages"]]
    result = qa.invoke({"question": prompt, "chat_history": chat_history_tuples})
    response = result["answer"]
    st.write("📄 Documentos Recuperados:")
    for doc in result["source_documents"]:
        st.write(f"🔹 {doc.page_content[:500]}")  # Muestra un fragmento de cada documento
        st.write(f"📌 Fuente: {doc.metadata.get('source', 'Desconocida')}")
    
    st.session_state["messages"].append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
