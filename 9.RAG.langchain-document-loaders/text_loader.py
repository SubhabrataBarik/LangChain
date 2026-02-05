## API
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from huggingface_hub.utils import HfHubHTTPError
import time
from dotenv import load_dotenv
load_dotenv()
from langchain.text_splitter import CharacterTextSplitter

## API
llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    max_new_tokens=256,
    temperature=0.7
)

## API
def invoke_with_retry(chain, inputs, retries=3, delay=5):
    for attempt in range(retries):
        try:
            return chain.invoke(inputs)
        except HfHubHTTPError as e:
            print(f"[Attempt {attempt+1}] Hugging Face Error: {e}")
            time.sleep(delay)
        except Exception as e:
            print(f"[Attempt {attempt+1}] Other Error: {e}")
            time.sleep(delay)
    print("⚠️ Unable to generate after multiple attempts.")
    return None


# ## LOCAL
# from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
# import os
# os.environ['HF_HOME'] = 'D:/PENDRIVE 32 GB/CHAT BOTS LOCAL/huggingface_cache'

# ## LOCAL
# llm = HuggingFacePipeline.from_model_id(
#     model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
#     task='text-generation',
#     pipeline_kwargs=dict(
#         temperature=1,
#         max_new_tokens=100
#     )
# )


from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


model = ChatHuggingFace(llm=llm)
# model = ChatOpenAI()

prompt = PromptTemplate(
    template='Write a summary for the following poem - \n {poem}',
    input_variables=['poem']
)

parser = StrOutputParser()

loader = TextLoader('9.langchain-document-loaders/cricket.txt', encoding='utf-8')
docs = loader.load()

# Split document into manageable chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = text_splitter.split_documents(docs)

print(type(docs))

print(len(docs))

print(docs[0].page_content)

print(docs[0].metadata)

chain = prompt | model | parser

# print(chain.invoke({'poem':docs[0].page_content}))

# Process each chunk
print(f"\n📄 Total Chunks: {len(split_docs)}")
all_summaries = []
for i, doc in enumerate(split_docs):
    print(f"\n🔹 Chunk {i+1}:")
    result = invoke_with_retry(chain, {'poem': doc.page_content})
    print(result)
    if result:
        all_summaries.append(result)
# Optionally join summaries together
final_summary = "\n".join(all_summaries)
print("\n📝 Final Combined Summary:\n")
print(final_summary)

# result = invoke_with_retry(chain, {'poem': docs[0].page_content})
# print("\n📝 Summary Result:")
# print(result)