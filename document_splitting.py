import os
import openai
import sys
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter, MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings
import numpy as np
from langchain_community.vectorstores import Chroma

sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

r'''
chunk_size = 26
chunk_overlap= 4

r_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
c_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

text1 = 'abcdefghijklmnopqrstuvwxyz'

#print(r_splitter.split_text(text1))

text2 = 'abcdefghijklmnopqrstuvwxyzabcdefg'

#print(r_splitter.split_text(text2))

text3 = "a b c d e f g h i j k l m n o p q r s t u v w x y z"

#print(c_splitter.split_text(text3))


some_text = """When writing documents, writers will use document structure to group content. \
This can convey to the reader, which idea's are related. For example, closely related ideas \
are in sentances. Similar ideas are in paragraphs. Paragraphs form a document. \n\n  \
Paragraphs are often delimited with a carriage return or two carriage returns. \
Carriage returns are the "backslash n" you see embedded in this string. \
Sentences have a period at the end, but also, have a space.\
and words are separated by space."""


#print(len(some_text))

c_splitter = CharacterTextSplitter(
    chunk_size=450,
    chunk_overlap = 0,
    separator=' '
)

r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0,
    separators=["\n\n", "\n","(?<=\.)", " ", ""]
)

#print(c_splitter.split_text(some_text))
#print(r_splitter.split_text(some_text))


loader = PyPDFLoader(r"C:\Users\alexl\OneDrive\Escritorio\MachineLearning-Lecture01.pdf")
pages = loader.load()

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1500,
    chunk_overlap=1000,
    length_function=len
)

docs = text_splitter.split_documents(pages)
#print(len(docs))
#print(len(pages))

text_splitter = TokenTextSplitter(chunk_size=1, chunk_overlap=0)

text1="foo bar bazzyfoo"
print(text_splitter.split_text(text1))



docs = text_splitter.split_documents(pages)
print(docs[0])
print(docs [0].metadata)

markdown_document = """# Title\n\n \
## Chapter 1\n\n \
Hi this is Jim\n\n Hi this is Joe\n\n \
### Section \n\n \
Hi this is Lance \n\n 
## Chapter 2\n\n \
Hi this is Molly"""


headers_to_split = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split,
)

md_headers_splits = markdown_splitter.split_text(markdown_document)
print(md_headers_splits[0])
print(md_headers_splits[1])

'''



loaders = [
    PyPDFLoader(r"C:\Users\alexl\OneDrive\Escritorio\MachineLearning-Lecture01.pdf"),
    PyPDFLoader(r"C:\Users\alexl\OneDrive\Escritorio\MachineLearning-Lecture01.pdf"),
    PyPDFLoader(r"C:\Users\alexl\OneDrive\Escritorio\MachineLearning-Lecture02.pdf"),
    PyPDFLoader(r"C:\Users\alexl\OneDrive\Escritorio\MachineLearning-Lecture03.pdf")
    ]

docs=[]

for loader in loaders:
    docs.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150
)

splits = text_splitter.split_documents(docs)
#print(len(splits))

embedding = OpenAIEmbeddings()

frase1= "Me gustan los perros"
frase2= "Me gustan los gatos"
frase3= "El tiempo parece feo"


embedding1= embedding.embed_query(frase1)
embedding2= embedding.embed_query(frase2)
embedding3= embedding.embed_query(frase3)

#print(np.dot(embedding1,embedding2))
#print(np.dot(embedding1,embedding3))
#print(np.dot(embedding2,embedding3))


persist_directory = r"C:\Users\alexl\OneDrive\Escritorio\chroma"

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)


#print(vectordb._collection.count())

question = "what did they say about matlab?"
docs = vectordb.similarity_search(question,k=5)

print(len(docs))

#print(docs[0].page_content)

vectordb.persist()

print(docs[0])
print(docs[1])



