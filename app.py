import os

import streamlit as st
from langchain_classic.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

PERSIST_DIR = "chroma_db"

st.set_page_config(page_title="Agente BimBam Buy", page_icon="🛍️")
st.title("🛍️ Agente de soporte — BimBam Buy")
st.caption("Preguntá sobre envíos, garantías, reembolsos, pagos o el programa de afiliados.")

# La API key se busca primero en st.secrets (Streamlit Cloud) y si no,
# en una variable de entorno (para correr local). st.secrets lanza una
# excepción si no existe ningún secrets.toml, así que la contenemos acá.
try:
    groq_api_key = st.secrets.get("GROQ_API_KEY")
except Exception:
    groq_api_key = None
groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    st.error(
        "Falta configurar GROQ_API_KEY. Localmente: variable de entorno. "
        "En Streamlit Cloud: Settings → Secrets."
    )
    st.stop()
os.environ["GROQ_API_KEY"] = groq_api_key


@st.cache_resource
def cargar_cadena():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(
        collection_name="bimbambuy_docs",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    # llama-3.3-70b-versatile fue deprecado por Groq (jun. 2026); usamos el reemplazo recomendado.
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)


qa_chain = cargar_cadena()

# Historial de la conversación
if "historial" not in st.session_state:
    st.session_state.historial = []

# Mostrar mensajes anteriores
if not st.session_state.historial:
    with st.chat_message("assistant"):
        st.markdown(
            "¡Hola! 👋 Preguntame sobre envíos, garantías, reembolsos, pagos "
            "o el programa de afiliados de BimBam Buy."
        )

for mensaje in st.session_state.historial:
    with st.chat_message(mensaje["role"]):
        st.markdown(mensaje["content"])
        if mensaje.get("fuentes"):
            st.caption("📄 Fuentes: " + ", ".join(mensaje["fuentes"]))

# Entrada de chat (queda fija abajo, como en Claude/ChatGPT)
pregunta = st.chat_input("Escribí tu pregunta sobre BimBam Buy...")

if pregunta:
    st.session_state.historial.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant"):
        with st.spinner("Buscando en los documentos..."):
            resultado = qa_chain.invoke({"query": pregunta})
        respuesta = resultado["result"]
        fuentes = sorted(
            {doc.metadata.get("documento", "desconocido") for doc in resultado["source_documents"]}
        )
        st.markdown(respuesta)
        st.caption("📄 Fuentes: " + ", ".join(fuentes))

    st.session_state.historial.append(
        {"role": "assistant", "content": respuesta, "fuentes": fuentes}
    )
