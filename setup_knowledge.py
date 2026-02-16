import os
import shutil
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- CONFIGURACIÓN ---
DATA_PATH = "data/raw"
VECTOR_DB_PATH = "data/vectors"

# AQUÍ DEFINIMOS QUÉ LEER. 
# Solo ponemos los archivos de CONTEXTO (Texto).
# La "Base_maestra_kinesis.csv" NO la ponemos aquí (esa la usará Pandas después).
FILES_TO_INDEX = [
    "Metodología Kinesis.pdf",
    "Explicacion_contexto_programa.csv" 
]

def main():
    print("🧠 INICIANDO PROCESO DE MEMORIZACIÓN...")

    # 1. Limpiar la memoria anterior si existe (para empezar limpio)
    if os.path.exists(VECTOR_DB_PATH):
        shutil.rmtree(VECTOR_DB_PATH)
        print("   -> Memoria vieja borrada. Creando nueva base de vectores...")

    all_docs = []

    # 2. Cargar los documentos de texto
    for filename in FILES_TO_INDEX:
        file_path = os.path.join(DATA_PATH, filename)
        
        if not os.path.exists(file_path):
            print(f"❌ ERROR: No encuentro '{filename}' en data/raw. Verifica el nombre.")
            continue
            
        print(f"   -> Leyendo y procesando: {filename}...")
        
        try:
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                docs = loader.load()
            elif filename.endswith(".csv"):
                # Para el CSV de contexto, leemos cada fila como un párrafo de texto
                loader = CSVLoader(file_path, encoding="utf-8")
                docs = loader.load()
            
            all_docs.extend(docs)
            print(f"      OK. Se extrajeron {len(docs)} páginas/filas.")
            
        except Exception as e:
            print(f"      ❌ Error leyendo el archivo: {e}")

    if not all_docs:
        print("❌ No hay nada que procesar. Abortando.")
        return

    # 3. Dividir en trozos (Chunks) para que la IA los pueda digerir
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Tamaño del trozo
        chunk_overlap=200 # Solapamiento para no cortar ideas
    )
    splits = text_splitter.split_documents(all_docs)
    print(f"   -> Texto fragmentado en {len(splits)} pedazos.")

    # 4. Crear los Vectores (Embeddings) y guardar en Disco
    print("   -> Generando vectores (Traduciendo texto a números)...")
    
    # Usamos un modelo ligero y gratuito que corre en tu CPU
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embedding_function,
        persist_directory=VECTOR_DB_PATH
    )
    
    print("✅ ¡ÉXITO! La memoria ha sido creada en la carpeta 'data/vectors'.")

if __name__ == "__main__":
    main()