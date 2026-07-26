# Agente de IA — BimBam Buy

Agente de preguntas y respuestas sobre la documentación interna de BimBam Buy (empresa
ficticia de e-commerce), construido con LangChain y desplegado con Streamlit. El agente
responde consultas en lenguaje natural sobre envíos, garantías, reembolsos, pagos y el
programa de afiliados, combinando información de varios documentos cuando la pregunta lo
requiere.

**App en vivo:** https://challengealuraagente-x42pwivsbkrdwnmqvxqr6i.streamlit.app/

## Arquitectura

```
PDFs (documentos/)
      │
      ▼
PyPDFLoader ──► chunks (RecursiveCharacterTextSplitter)
      │
      ▼
Embeddings locales (HuggingFace, sentence-transformers/all-MiniLM-L6-v2)
      │
      ▼
Vector store persistente (Chroma, ./chroma_db)
      │
      ▼
Retriever (top-k=5) ──► RetrievalQA (LangChain) ──► LLM (Groq, openai/gpt-oss-120b)
      │
      ▼
Interfaz de chat (Streamlit)
```

**Por qué estas decisiones:**
- **Embeddings locales** en vez de un proveedor externo: evita límites de cuota y no
  requiere API key para esa parte del pipeline.
- **Groq** para el LLM de chat: inferencia rápida y gratuita dentro de límites generosos.
- **Chroma persistente**: el índice se calcula una sola vez (`ingest.py`) y se sube ya
  generado al repo, para que la app no tenga que recalcular embeddings en cada arranque.

## Documentos utilizados

1. Guía de Tiempos y Costos de Envío
2. Manual de Garantía de Productos
3. Política de Reembolsos y Devoluciones
4. Preguntas Frecuentes sobre Métodos de Pago
5. Programa de Afiliados

Estos documentos están interconectados (cada uno referencia a los demás en su sección de
"coordinación con otros documentos"), lo que permite probar preguntas que cruzan varias
fuentes a la vez.

## Ejemplos de preguntas y respuestas

**Pregunta:** ¿Un producto llega dañado y quiero devolverlo, tiene costo para mí y cuánto
tarda el reembolso?

**Respuesta:** No tiene costo si se reporta dentro de las 48 horas con evidencia suficiente
(fotos o video). El reembolso se procesa al mismo medio de pago y suele reflejarse entre 5
y 10 días hábiles tras la aprobación.
*(Fuentes: Manual de Garantía de Productos, Política de Reembolsos y Devoluciones,
Preguntas Frecuentes sobre Métodos de Pago)*

**Pregunta:** Si un afiliado refirió una venta y el cliente pide reembolso, ¿qué pasa con
la comisión?

**Respuesta:** Depende del tipo de reembolso: uno total anula la comisión por completo; uno
parcial la ajusta proporcionalmente al monto neto pagado; un cambio por producto
equivalente o una resolución de garantía sin devolución de dinero generalmente mantiene la
comisión, siempre que la venta siga confirmada.
*(Fuentes: Política de Reembolsos y Devoluciones, Programa de Afiliados)*

## Cómo ejecutar el proyecto localmente

Requisitos: Python 3.10+ y una API key gratuita de [Groq](https://console.groq.com).

```bash
git clone https://github.com/Malcuthrad/Challenge_Alura_Agente.git
cd Challenge_Alura_Agente

python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt

# Variable de entorno con tu API key de Groq
$env:GROQ_API_KEY = "tu_api_key"     # PowerShell
# export GROQ_API_KEY="tu_api_key"  # Linux/Mac

# Solo la primera vez (o si cambian los documentos):
python ingest.py

streamlit run app.py
```

La app queda disponible en `http://localhost:8501`.

## Estructura del repositorio

```
├── documentos/       # PDFs fuente
├── chroma_db/        # índice vectorial ya calculado (persistente)
├── ingest.py         # Etapa 1: carga, chunking y generación del índice
├── app.py            # Etapa 2: agente de Q&A + interfaz de chat
└── requirements.txt
```

## Deploy

Desplegado en [Streamlit Community Cloud](https://streamlit.io/cloud), conectado
directamente a este repositorio. La única configuración necesaria en
**Settings → Secrets** es:

```toml
GROQ_API_KEY = "tu_api_key"
```
