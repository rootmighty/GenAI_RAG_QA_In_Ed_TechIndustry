from langchain_community.llms import OpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader, UnstructuredURLLoader
import langchain
import langchain_community
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

#Loading Environment Variables
load_dotenv()

llm = OpenAI(temperature = 0.1, max_tokens = 500)

#Loading csv data

csv_file_path = "codebasics_faqs.csv"

#Creating embeddings and vector database
embeddings = OpenAIEmbeddings()


# Save the FAISS index to a file
vectordb_file_path = "faiss_store_openai"

#We will savec our vectordb to file
def create_vector_db():
    csv_loader = CSVLoader(file_path=csv_file_path, source_column='prompt', encoding="utf-8")
    docs = csv_loader.load()
    vectordb = FAISS.from_documents(documents=docs, embedding=embeddings)
    vectordb.save_local(vectordb_file_path)


#The Q&A chain
def get_qa_chain():
    #Load the vecor db from the local folder
    vectordb = FAISS.load_local(vectordb_file_path, embeddings=embeddings, allow_dangerous_deserialization=True)

    #Creating my retriever
    retriever = vectordb.as_retriever()

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}
    
    ANSWER:"""


    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}

    chain = RetrievalQA.from_chain_type(llm=llm,
                                chain_type="stuff",
                                retriever=retriever,
                                input_key="query",
                                return_source_documents=True,
                                chain_type_kwargs=chain_type_kwargs)
    
    return chain
    
if __name__ == "__main__":
    chain = get_qa_chain()

    print(chain("Do you provide blockchain course"))
