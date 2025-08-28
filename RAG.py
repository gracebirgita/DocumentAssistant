import os
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

load_dotenv()



# 1. input file/text as basis info
print("Input your information resource: ")
document_input = input()

# 2. chunking
def chunk_text(text, chunk_size=80):
    words=text.split()
    return [
        {'content': ' '.join(words[i:i+chunk_size]), 'source': f'chunk-{i//chunk_size+1}'}
        for i in range(0, len(words), chunk_size)
    ]
    # return[
    #     ' '.join(words[i:i+chunk_size])
    #     for i in range(0, len(words), chunk_size)
    # ]

documents = chunk_text(document_input, chunk_size=80)
print(documents)

# 3. retrieve info
# sentence tranasformer
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# -> vector embedding ([0.01.., -..., ...., ...])
def get_embedding(text):
    return embedder.encode(text)

def cosine_sim(a,b):
    # multiply 2 vector / (len norm a /len norm b)
    return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve(query, docs, top_k=2):
    query_emb = get_embedding(query)
    chunk_embs = [get_embedding(chunk['content']) for chunk in docs]
    scored_chunks=[]

    for chunk, emb in zip(docs, chunk_embs):
        score = cosine_sim(query_emb, emb)
        scored_chunks.append((score, chunk))

    scored_chunks.sort(reverse=True)
    return [chunk for score, chunk in scored_chunks[:top_k] if score>0]
    # input user -> by words appear
    # query_words= set(query.lower().split())
    # scored_chunks=[]
    # for chunk in docs:
    #     chunk_words = set(chunk.lower().split())
    #     score = len(query_words & chunk_words) # same amount words
    #     scored_chunks.append((score, chunk))
    #     print(chunk)
    #     print(score)
    # # sort by highest score
    # scored_chunks.sort(reverse=True)
    # # get top_k chunk relevant
    # return[chunk for score, chunk in scored_chunks[:top_k] if score>0]

question = str(input("Input your question about docs: "))

# 4. builld prompt
relevant_chunks = retrieve(question, documents, top_k=2)
# combine context & question
# context = '\n'.join(relevant_chunks)
context = '\n'.join([chunk['content'] for chunk in relevant_chunks])
sources =[(chunk['source'], chunk['content']) for chunk in relevant_chunks]

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key= os.getenv('OPENROUTER_API_KEY'),
)
print("API KEY:", os.getenv('OPENROUTER_API_KEY'))

completion = client.chat.completions.create(
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

print('Source')
for source, content in sources:
    print(f"- {content} [{source}]")

# from langchain_community.document_loaders import WebBaseLoader

# loader = WebBaseLoader(web_paths=["https://www.example.com"])
# docs = loader.load()
# print(docs[0].page_content)  # Menampilkan isi halaman

# document_input = docs[0].page_content
# documents = chunk_text(document_input, chunk_size=80)

# from langchain_community.document_loaders import TextLoader
# loader = TextLoader("namafile.txt")
# docs = loader.load()
# document_input = docs[0].page_content
# documents = chunk_text(document_input, chunk_size=80)