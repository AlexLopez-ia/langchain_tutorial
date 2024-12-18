import os
import openai
import sys
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.document_loaders import WebBaseLoader

sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

openai.api_key = os.environ['OPENAI_API_KEY']


#leer pdf

loader = PyPDFLoader("C:\\Users\\alexl\\OneDrive\\Escritorio\\UNI_Asignaturas\\3 Año ing\\BDII\\P1_Apuntes_DDL.pdf")
pages = loader.load()
len(pages)

page = pages[0]
print(page.page_content[:500])


#Leer youtibe videos
#url = "https://www.youtube.com/watch?v=FOm3WknWW6s&t=2s"
#save_dir = "C:\\Users\\alexl\\OneDrive\\Escritorio\\Youtube"
#loader = GenericLoader(YoutubeAudioLoader([url], save_dir), OpenAIWhisperParser)
#docs = loader.load()

#docs[0].page_content[0:500]


#Leer urls

