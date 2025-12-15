# imports
import os
import shutil
from dotenv import load_dotenv
import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# env variables
load_dotenv()

csv_file = os.getenv("REVIEWS")
chroma_path = os.getenv("CHROMA_PATH")
api_key = os.getenv("OPENAI_KEY")

# funções básicas de ingestão de dados
def limpar_base():
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)
        print("Banco antigo limpo.")

def ler_csv(caminho):
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Arquivo CSV não encontrado em: {caminho}")
    df = pd.read_csv(caminho)

    if "Review" not in df.columns:
        raise ValueError("Não há coluna chamada Review no csv.")
    
    docs = []
    for i, texto in enumerate(df["Review"]):
        docs.append(Document(page_content=str(texto), metadata={"id": i}))
    return docs

def criar_base(documentos):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Chroma.from_documents(
        documents=documentos,
        embedding=embeddings,
        persist_directory=chroma_path,
        collection_name="ifood_reviews"
    )
    print(f"{len(documentos)} anexados.")

# main
def main():
    limpar_base()
    documentos = ler_csv(csv_file)
    criar_base(documentos)

if __name__ == "__main__":
    main()