#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pickle
import pandas as pd
from tqdm import tqdm
from itertools import product
from langchain.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer


# In[1]:


dir = '../data/'
train = pd.read_csv(dir + "train.csv")
list(product([f"질문_{x}" for x in range(1,3)],[f"답변_{x}" for x in range(1,6)]))

train_data = []

for q,a in list(product([f"질문_{x}" for x in range(1,3)],[f"답변_{x}" for x in range(1,6)])):
    for i in range(len(train)):
        train_data.append(
            "질문: "+ train.at[i,q] + " 답변 : " + train.at[i,a]
        )


# In[2]:


len(train_data)


# In[3]:


pd.DataFrame(train_data).to_csv(dir + "train_data.csv",index=False,encoding='utf-8')
loader = CSVLoader(file_path=dir + 'train_data.csv',encoding='utf-8')
data = loader.load()


# In[5]:


modelPath = "distiluse-base-multilingual-cased-v1"

model_kwargs = {'device':'cuda'}

encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


# In[16]:


db = FAISS.from_documents(data, embedding=embeddings)


# In[17]:


db.save_local(dir + "faiss_index")


# ## Load FAISS file

# In[6]:


db = FAISS.load_local(dir + "faiss_index", embeddings)


# In[ ]:





# In[22]:


retriever = db.as_retriever(search_kwargs={"k": 4})


# In[27]:


from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import LlamaForCausalLM, AutoTokenizer, pipeline

model_id = "beomi/llama-2-ko-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = LlamaForCausalLM.from_pretrained(model_id).to(0)


# In[26]:


pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512,device=0)
hf = HuggingFacePipeline(pipeline=pipe)


# In[ ]:


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# In[ ]:


template = """마지막에 질문에 답하려면 다음과 같은 맥락을 사용합니다.

{context}

질문: {question}

유용한 답변:"""
custom_rag_prompt = PromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | hf
    | StrOutputParser()
)


# In[ ]:


for chunk in rag_chain.stream("도배지에 녹은 자국이 발생하는 주된 원인과 그 해결 방법은 무엇인가요?"):
    print(chunk, end="", flush=True)


# In[ ]:


test = pd.read_csv("test.csv")


# In[ ]:


test


# In[ ]:


result = []



for i in tqdm(range(len(test))):
  _id = test.at[i,'id']
  _q = test.at[i,'질문']
  _a = []
  for chunk in rag_chain.stream(_q):
      _a.append(chunk)
      print(chunk, end="", flush=True)
  result.append(
      {
          "id":_id,
          "대답":" ".join(_a)
      }
  )
  print()


# In[ ]:


with open("result.pkl",'wb') as f:
  pickle.dump(result,f)


# In[ ]:


_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')


# In[ ]:


for i in range(len(result)):
  result[i]['embedding'] = _model.encode(result[i]['대답'].replace("\u200b"," "))


# In[ ]:


submission = []




for i in range(len(result)):
  tmp = {"id":result[i]['id'],}
  for j in range(len(result[i]['embedding'])):
    tmp[f'vec_{j}'] = result[i]['embedding'][j]
  submission.append(
      tmp
  )


# In[ ]:


pd.DataFrame(submission).to_csv("submission_RAG.csv",index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




