from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
import os
from dotenv import load_dotenv
from openai import OpenAI
# import getpass

load_dotenv()
def main():
    print("1. uplaod link")
    print("2. uplad pdf")
    print("3. input text")
    input_format=str(input(">> "))

    # langchain loader
    match input_format:
        case "1":
            url = input("Input URL: ")
            loader = WebBaseLoader(web_paths= [url])
            docs = loader.load()
        case "2":
            filename= input("Input file name: ")
            loader= TextLoader(input_format)
            docs = loader.load()
        case "3":
            text = input("Input text: ")
            docs= [Document(page_conttent=text)]
        case _:
            print("option invalid")
            return

    
    # 2. split doc -> chunk (langchain)
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
    splitter_docs = splitter.split_documents(docs)
    # 3. embedding & vectorstore (langchain)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splitter_docs, embeddings)
    # 4. retriever & QA chain(langchain)
    retriever = vectorstore.as_retriever()

    question = input("Insert your question: ")
    relevant_docs = retriever.get_relevant_documents(question)
    context = '\n'.join([doc.page_content for doc in relevant_docs])
    print("RETRIEVER: ", retriever)
    print("CONTEXT: ", context)

    # OpenAI wrapper for OpenRouter
    llm = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv('OPENROUTER_API_KEY'),
    )
    print("API KEY:", os.getenv('OPENROUTER_API_KEY'))

    completion = llm.chat.completions.create(
    model="deepseek/deepseek-r1-0528-qwen3-8b:free",
    messages=[
        {
        "role": "user",
        "content": (
            f"You are an expert assistant. "
            f"Given the following context, answer the question as accurately and concisely as possible. "
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            f"Answer:"
            )
        }
    ]
    )

    # 5. generate
    print("Response : ")
    print(completion.choices[0].message.content)

    # get resource snippet
    print("SOURCES:")
    for i,doc in enumerate(relevant_docs,1):
        print(f"{i}. doc.page_content")


    # qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    # 5. generate answer
    # result = qa_chain.invoke({"query": question})
    # print("Jawaban:", result["result"])

if __name__=="__main__":

    main()
