from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import faiss
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

import os

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your openAi key")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "Your Langchain Key")

pdfReader=PdfReader('syllabus.pdf')

from typing_extensions import Concatenate
raw_text=''
for i , page in enumerate(pdfReader.pages):
    content=page.extract_text()
    if content:
        raw_text+=content

textsplitter= CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,

)
texts=textsplitter.split_text(raw_text)
embedding=OpenAIEmbeddings()

document_search= FAISS.from_texts(texts,embedding)

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import openai

chain= load_qa_chain(openai(),chain_type="stuff")
query= "business intelligence"
docs= document_search.similarity_search(query)
chain.run(input_documents=docs, question=query)

