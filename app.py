import streamlit as st
# summarizer
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
# load file reader
import PyPDF2
from langchain_community.document_loaders import WebBaseLoader
from sentence_transformers import SentenceTransformer

# RAG
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain_core.documents import Document
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
print(np.__version__)
st.set_page_config(page_title="Document Assistant", layout="wide")
# prompt = st.chat_input(
#     "Say something and/or attach an image",
#     accept_file=True,
#     file_type=["jpg", "jpeg", "png"],
# )
# if prompt and prompt.text:
#     st.markdown(prompt.text)
# if prompt and prompt["files"]:

load_dotenv()

def extract_text_from_file(file_or_url, input_type=None):
    # link
    if input_type=="link":
        if not file_or_url or not isinstance(file_or_url, str) or file_or_url.strip()=="":
            return ""
        # url = input("Input URL: ")
        loader = WebBaseLoader(web_paths= [file_or_url])
        docs = loader.load()
        return "\n".join([doc.page_content for doc in docs])
    elif hasattr(file_or_url, "type"):
        if file_or_url.type=="text/plain":
            return file_or_url.getvalue().decode("utf-8")
        # # pdf
        elif file_or_url.type =="application/pdf":
            pdf_reader = PyPDF2.PdfReader(file_or_url)
            text=""
            for page in pdf_reader.pages:
                text+=page.extract_text() or ""
            return text
    elif isinstance(file_or_url, str):
        return file_or_url
    return ""
        

# summarizer = pipeline('summarization', model='facebook/bart-large-cnn')
# summarizer = pipeline(
#     "summarization",
#     model="facebook/bart-large-cnn",
#     device=-1,                  # force CPU
#     torch_dtype="float32"       # not half/quantized
# )
# @st.cache_resource
# def load_summarizer():
#     model_candidates = [
#         "facebook/bart-large-cnn",          # main
#         "sshleifer/distilbart-cnn-12-6",    # fallback 1 (smaller)
#         "t5-small"                          # fallback 2 (most light)
#     ]

#     for model_name in model_candidates:
#         try:
#             tokenizer = AutoTokenizer.from_pretrained(model_name)

#             model = AutoModelForSeq2SeqLM.from_pretrained(
#                 model_name,
#                 torch_dtype=torch.float32,  
#                 device_map=None        
#             )

#             summarizer = pipeline(
#                 "summarization",
#                 model=model,
#                 tokenizer=tokenizer,
#                 device=-1  # -1 = CPU only
#             )
#             print(f"Loaded summarizer: {model_name}")
#             return summarizer

#         except Exception as e:
#             print(f"Failed to load {model_name}: {e}")
#             continue

#     st.error(" No summarization model could be loaded.")
#     return None


# summarizer = load_summarizer()
# device = 0 if torch.cuda.is_available() else -1
# summarizer = pipeline(
#     "summarization",
#     model="facebook/bart-large-cnn",
#     device=device,                  # force CPU
#     torch_dtype=torch.float32       # not half/quantized
# )
device = 0 if torch.cuda.is_available() else -1
model_name = "facebook/bart-large-cnn"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model langsung ke CPU/GPU, no meta device
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,       # full precision (no half/quantized)
    low_cpu_mem_usage=False,         # disable meta init
    device_map=None                  # force no accelerate/meta
)

# Kalau ada GPU → pindah manual
if device >= 0:
    model = model.to(f"cuda:{device}")
else:
    model = model.to("cpu")

# Buat pipeline
summarizer = pipeline(
    "summarization",
    model=model,
    tokenizer=tokenizer,
    device=device
)
def summarize(text):
    if not text or len(text.strip())==0:
        st.write("please insert your text...")
        return ""
    
    if len(text.split())<10:
        return text 
    
    try: 
        summary_abs = summarizer(text, max_length=500,
                                min_length=10,
                                do_sample=False) # greedy search(token highest probability) - output stabil(sama)
        result_abs=summary_abs[0]['summary_text']
        # print(result_abs)
        return result_abs
    except Exception as e:
        st.error(f"Summarization error: {e}")
        return text
    

# RAG
def setup_rag_from_text(text):
    # process text & prepare RAG retriever
    # text split, create embeddings, store in vector

    docs = [Document(page_content=text)]
     # 2. split doc -> chunk (langchain)
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    splitter_docs = splitter.split_documents(docs)
    # 3. embedding & vectorstore (langchain)
    # model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    # model.to(dtype=torch.float32)
    # embeddings = HuggingFaceEmbeddings(model=model)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    vectorstore = FAISS.from_documents(splitter_docs, embeddings)
    # 4. retriever & QA chain(langchain)
    retriever = vectorstore.as_retriever()

    return retriever

def get_rag_response(retriever, question):
    # get response from RAG pipeline
    relevant_docs = retriever.get_relevant_documents(question)
    context = '\n'.join([doc.page_content for doc in relevant_docs])
    print("RETRIEVER: ", retriever)
    print("CONTEXT: ", context)

    # LLM
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
            f"Given the following context, answer the question as accurately and concisely as possible."
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            f"Answer:"
            )
        }
    ]
    )

    # 5. generate
    print("Response : ")
    response = completion.choices[0].message.content
    print(response)
    
    # get resource snippet
    # print("SOURCES:")
    # for i,doc in enumerate(relevant_docs,1):
    #     print(f"{i}. doc.page_content")
    sources=[doc.page_content for doc in relevant_docs]

    return response, sources


st.sidebar.title("-- Menu --")
app_mode = st.sidebar.selectbox(
    "Choose Menu",
    ['Summarizer', 'RAG Chatbot']
)

def main():
    if app_mode=="Summarizer":
        st.title("Document Summarizer")
        st.markdown("Upload file or insert text to get the summarize (in english).")

        input_method = st.radio(
            "Input format: ",
            ("Upload File", "Input Text")
        )
        text_to_summarize=""

        st.markdown("")

        if input_method=="Upload File":
            uploaded_file = st.file_uploader("Choose file (.txt, .pdf)", type=["txt","pdf"])
            if uploaded_file is not None:
                with st.spinner("Extract text from file..."):
                    text_to_summarize=extract_text_from_file(uploaded_file)
                    st.text_area("extracted text: ", text_to_summarize, height=200,disabled=True)
        elif input_method=="Input Text":
            text_to_summarize = st.text_area("Input your text (max 500 character):", height=500)
        
        # button summarize
        if st.button("Summarize"):
            if text_to_summarize:
                with st.spinner("Summarize the text..."):
                    summary = summarize(text_to_summarize)
                    if summary != text_to_summarize:
                        st.subheader("Summary: ")
                        # st.write(summary)
                        st.success(summary)
                    if summary ==text_to_summarize:
                        st.info("Text min: 10 character | max: 500 character")
            else:
                st.warning("Please insert your text...")

    # CHATBOT RAG
    elif app_mode =="RAG Chatbot":
        st.title("RAG Chatbot - ask from your document")
        st.markdown("Upload your PDF or text and ask the question (in english)")
        rag_input = st.radio(
            "Document for analysis: ",
            ("Upload File", "Input Text", "Insert Link")
        )

        st.write("")
        st.markdown("---")
        st.write("")
        if 'rag_ready' not in st.session_state:
                st.session_state.rag_ready=False
        if "messages" not in st.session_state: # history chat init
            st.session_state.messages=[]

        # by document PDF
        if rag_input=="Upload File":
            st.subheader("Upload your PDF")
            uploaded_pdf = st.file_uploader("Choose PDF file", type="pdf")
            full_text = extract_text_from_file(uploaded_pdf)
        # by directly text input
        elif rag_input=="Input Text":
            st.header("Input your resource text")
            full_text = st.text_area("Input your text:", height=250)
            # docs= [Document(page_content=full_text)]
        elif rag_input=="Insert Link":
            st.header("Input your resource link")
            url_resource = st.text_area("Input url:", height=68)
            full_text = extract_text_from_file(url_resource, input_type="link")

        # Simpan rag_input sebelumnya di session_state
        if "last_rag_input" not in st.session_state:
            st.session_state.last_rag_input = rag_input

        # Jika rag_input berubah, reset chat dan retriever
        if rag_input != st.session_state.last_rag_input:
            st.session_state.messages = []
            st.session_state.rag_ready = False
            if "retriever" in st.session_state:
                del st.session_state.retriever
            st.session_state.last_rag_input = rag_input

        save_clicked = st.button("Save")
        if save_clicked and full_text:
            with st.spinner("Extract text from file..."):
                # extract text & build vector store
                st.text_area("extracted text: ", full_text, height=200,disabled=True)
                st.session_state.retriever = setup_rag_from_text(full_text) # save retriever in session
                st.session_state.rag_ready = True
                st.session_state.messages = [] # Reset history chat
                st.success("Document sucess loaded!")
        elif save_clicked and not full_text:
            st.warning("please insert your document to search specific question..")

                        
                        
        ### INPUT QUESTION USER
        st.write("")
        st.markdown("---")
        st.write("")

        if st.session_state.rag_ready:
            st.header("Insert your question")
            # show history chat
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # input user
            if prompt:=st.chat_input("send question about your document..."):
                # add user messages -> history
                st.session_state.messages.append({"role":"user", "content":prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # give RAG response from chatbot
                with st.chat_message("assistant"):
                    print("process to search answer..")
                    with st.spinner("Search answer..."):
                        response, sources = get_rag_response(st.session_state.retriever, prompt)
                        st.markdown(response)
                        print("response process")
                        if sources:
                            st.markdown("**Sumber context:**")
                            for i,src in enumerate(sources, 1):
                                st.info(f"{i} - {src[:300]}...")
                        st.session_state.messages.append({"role":"assistant", "content": response})


if __name__=='__main__':

    main()










