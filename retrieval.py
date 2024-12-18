import os
import openai
import sys
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter, MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings
import numpy as np
from langchain_chroma import Chroma
from langchain_openai import OpenAI, ChatOpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo 
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

persist_directory = r"C:\Users\alexl\OneDrive\Escritorio\chroma"

embedding = OpenAIEmbeddings()
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

''''
try:
    print(f"Número de documentos en la colección: {vectordb._collection.count()}")
except Exception as e:
    print(f"Error al acceder a la colección: {e}")


texts = [
    """The Amanita phalloides has a large and imposing epigeous (aboveground) fruiting body (basidiocarp).""",
    """A mushroom with a large fruiting body is the Amanita phalloides. Some varieties are all-white.""",
    """A. phalloides, a.k.a Death Cap, is one of the most poisonous of all known mushrooms.""",
]


smalldb = Chroma.from_texts(texts, embedding=embedding)

question = "Tell me about all-white mushrooms with large fruiting bodies"

print(smalldb.similarity_search(question, k=2))

print(smalldb.max_marginal_relevance_search(question,k=2, fetch_k=3))

question = "what did they say about matlab?"
docs_ss = vectordb.similarity_search(question,k=3)


try:
    # Inicializar embeddings
    embedding = OpenAIEmbeddings()
    print("Embeddings inicializados")

    # Crear/cargar la base de datos vectorial
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )
    print("Base de datos vectorial cargada")

    # Realizar la búsqueda
    question = "what did they say about regression in the third lecture?"
    print(f"\nRealizando búsqueda para: '{question}'")

    docs = vectordb.similarity_search(
        question,
        k=3
    )
    
    print(f"\nSe encontraron {len(docs)} documentos")
    
    for i, doc in enumerate(docs, 1):
        print(f"\nDocumento {i}:")
        print(f"Metadatos: {doc.metadata}")
        print(f"Primeros 200 caracteres del contenido: {doc.page_content[:200]}...")

except Exception as e:
    print(f"Error durante la ejecución: {e}")
    print(f"Tipo de error: {type(e)}")

'''
   # Crear/cargar la base de datos vectorial
vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )

metadata_field_info = [
    AttributeInfo(
        name="source",
        description="The lecture the chunk is from, should be one of `docs/cs229_lectures/MachineLearning-Lecture01.pdf`, `docs/cs229_lectures/MachineLearning-Lecture02.pdf`, or `docs/cs229_lectures/MachineLearning-Lecture03.pdf`",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="The page from the lecture",
        type="integer",
    ),
]

document_content_description = "Lecture notes"
llm = OpenAI(temperature=0)
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectordb,
    document_content_description,
    metadata_field_info,
    verbose=True
)

question = "what did they say about regression in the third lecture"
docs = retriever.invoke(question)
print(docs)

for d in docs:
    print(d.metadata)



def pretty_print_docs(docs):
        print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
compressor = LLMChainExtractor.from_llm(llm)

# Corregir la inicialización del retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,  # Corregido de base_compression a base_compressor
    base_retriever=vectordb.as_retriever(search_type="mmr")  # Añadidos los paréntesis para llamar al método
)

question = "what did they say about matlab"
compressed_docs = compression_retriever.invoke(question)

# Corregir la función pretty_print_docs
def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))
    
# Llamar a la función correctamente
pretty_print_docs(compressed_docs)
