"""
Etapa 1: lectura y procesamiento de los documentos.

Uso:
    1. Colocá los 5 PDFs de BimBam Buy dentro de la carpeta ./documentos
    2. Configurá la variable de entorno GROQ_API_KEY
    3. Corré:  python ingest.py

Esto genera la carpeta ./chroma_db con el índice ya calculado,
para que app.py no tenga que recalcular embeddings en cada arranque.
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR = Path("documentos")
PERSIST_DIR = "chroma_db"


def cargar_documentos():
    all_docs = []
    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"No se encontraron PDFs en {DOCS_DIR.resolve()}. "
            "Copiá ahí los 5 documentos de BimBam Buy antes de correr este script."
        )
    for pdf_path in pdfs:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for page in pages:
            page.metadata["documento"] = pdf_path.name
        all_docs.extend(pages)
        print(f"  - {pdf_path.name}: {len(pages)} páginas")
    return all_docs


def main():
    print("Cargando documentos...")
    docs = cargar_documentos()
    print(f"Total de páginas cargadas: {len(docs)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Chunks generados: {len(chunks)}")

    print("Generando embeddings locales (HuggingFace) y guardando en Chroma...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="bimbambuy_docs",
        persist_directory=PERSIST_DIR,
    )
    print(f"Listo. Índice persistido en ./{PERSIST_DIR}")


if __name__ == "__main__":
    main()
