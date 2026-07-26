import html
import os

import streamlit as st
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

PERSIST_DIR = "chroma_db"

st.set_page_config(page_title="Agente BimBam Buy", page_icon="🛍️", layout="centered")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .bbb-header {
        background: linear-gradient(120deg, #7C5CFC 0%, #FF6B6B 100%);
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(124, 92, 252, 0.25);
    }
    .bbb-header h1 {
        font-family: 'Baloo 2', sans-serif;
        color: white;
        font-size: 2rem;
        margin: 0 0 6px 0;
    }
    .bbb-header p {
        color: rgba(255, 255, 255, 0.9);
        margin: 0;
        font-size: 0.95rem;
    }

    </style>

    <div class="bbb-header">
        <h1>🛍️ Agente de soporte — BimBam Buy</h1>
        <p>Pregunte sobre envíos, garantías, reembolsos, pagos o el programa de afiliados.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

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


PROMPT_TEMPLATE = """Sos el agente de soporte de BimBam Buy. Respondé la pregunta del cliente
usando exclusivamente la información del contexto de abajo.

Estilo de respuesta:
- Tono cercano, claro y natural, como si le hablaras directamente a la persona.
- Nunca uses tablas ni encabezados (nada de ###, ni títulos en negrita tipo sección).
- Escribí en párrafos cortos. Usá una lista simple con guiones solo si hay pasos concretos
  a seguir, y no más de 4-5 puntos.
- Priorizá lo esencial (qué hacer, plazos, costos) sin repetir el mismo contenido en
  distintos formatos ni agregar un "resumen" después de ya haber explicado todo.
- No inventes datos de contacto (teléfonos, chats en vivo, emails) que no estén
  explícitamente en el contexto.
- Si el contexto no cubre la pregunta, decilo claramente en vez de inventar una respuesta.

Contexto:
{context}

Pregunta del cliente: {question}

Respuesta:"""

QA_PROMPT = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])


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
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT},
    )


qa_chain = cargar_cadena()

# Historial de la conversación
if "historial" not in st.session_state:
    st.session_state.historial = []

AVATAR_ASISTENTE = "🛍️"
AVATAR_USUARIO = "🙂"


def mostrar_burbuja(role, contenido, fuentes=None):
    es_usuario = role == "user"
    avatar = AVATAR_USUARIO if es_usuario else AVATAR_ASISTENTE
    fondo = "#FFF1EE" if es_usuario else "#F0EBFF"
    direccion = "row-reverse" if es_usuario else "row"
    margen = "margin-left:18%;" if es_usuario else "margin-right:18%;"
    alineacion_texto = "right" if es_usuario else "left"

    contenido_html = html.escape(contenido).replace("\n", "<br>")
    fuentes_html = ""
    if fuentes:
        fuentes_html = (
            f'<div style="font-size:0.8rem;color:#6b7280;margin-top:8px;">'
            f'📄 Fuentes: {html.escape(", ".join(fuentes))}</div>'
        )

    partes = [
        f'<div style="display:flex;flex-direction:{direccion};align-items:flex-start;'
        f'gap:10px;{margen}margin-bottom:16px;">',
        f'<div style="font-size:1.5rem;line-height:1;">{avatar}</div>',
        f'<div style="background-color:{fondo};border-radius:16px;padding:10px 16px;'
        f'max-width:80%;text-align:{alineacion_texto};">',
        f'<div>{contenido_html}</div>',
        fuentes_html,
        '</div>',
        '</div>',
    ]
    st.markdown("".join(partes), unsafe_allow_html=True)


# Mostrar historial (o mensaje de bienvenida si todavía no hay nada)
if not st.session_state.historial:
    mostrar_burbuja(
        "assistant",
        "¡Hola! 👋 Preguntame sobre envíos, garantías, reembolsos, pagos "
        "o el programa de afiliados de BimBam Buy.",
    )

for mensaje in st.session_state.historial:
    mostrar_burbuja(mensaje["role"], mensaje["content"], mensaje.get("fuentes"))

# Entrada de chat (queda fija abajo, como en Claude/ChatGPT)
pregunta = st.chat_input("Escribí tu pregunta sobre BimBam Buy...")

if pregunta:
    st.session_state.historial.append({"role": "user", "content": pregunta})
    mostrar_burbuja("user", pregunta)

    with st.spinner("Buscando en los documentos..."):
        resultado = qa_chain.invoke({"query": pregunta})
    respuesta = resultado["result"]
    fuentes = sorted(
        {doc.metadata.get("documento", "desconocido") for doc in resultado["source_documents"]}
    )
    mostrar_burbuja("assistant", respuesta, fuentes)

    st.session_state.historial.append(
        {"role": "assistant", "content": respuesta, "fuentes": fuentes}
    )
