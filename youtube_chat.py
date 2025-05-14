from dotenv import load_dotenv,find_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import textwrap 
from langchain_core.output_parsers import StrOutputParser


load_dotenv(find_dotenv())
embeddings = OpenAIEmbeddings()

def create_db_from_youtube_video_url(video_url):
    loader = YoutubeLoader.from_youtube_url(video_url)  
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
    docs = text_splitter.split_documents(transcript)

    db = Chroma.from_documents(docs,embeddings)
    return db


#OpenAI + Langchain Prompt Engineering
chat = ChatOpenAI(model="gpt-3.5-turbo",temperature=2.0)

template = """
    You are a helpful assistant that that can answer questions about youtube videos 
    based on the video's transcript: {docs}
    
    Only use the factual information from the transcript to answer the question.
    
    If you feel like you don't have enough information to answer the question, say "I don't know".
    
    """
system_message_prompt = SystemMessagePromptTemplate.from_template(template)

human_message = "Answer the following question:{question}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_message)

#This is work without .from_messages too.
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])

chain = chat_prompt | chat | StrOutputParser()


#Final Function
def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])
    
    response = chain.invoke({"question": query, "docs": docs_page_content})
    return response



#Example Usage
video_url = "https://www.youtube.com/watch?v=aywZrzNaKjs"
db = create_db_from_youtube_video_url(video_url)

query = "What is the video about?"
response = get_response_from_query(db,query)
print(textwrap.fill(response,width=85))












