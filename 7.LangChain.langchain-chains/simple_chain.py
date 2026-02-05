## API
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from huggingface_hub.utils import HfHubHTTPError
import time

## API ~ TinyLlama/TinyLlama-1.1B-Chat-v1.0 ~ google/gemma-2-2b-it
llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)

## API
def invoke_with_retry(chain, input_dict, retries=3, delay=5):
    for attempt in range(retries):
        try:
            return chain.invoke(input_dict)
        except HfHubHTTPError as e:
            print(f"[Attempt {attempt + 1}] HF Hub Error: {e}")
            time.sleep(delay)
        except Exception as e:
            print(f"[Attempt {attempt + 1}] Other Error: {e}")
            time.sleep(delay)
    print("❌ Failed to get a response after retries.")
    return None


## LOCAL
# from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
# from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# import os
# os.environ['HF_HOME'] = 'D:/PENDRIVE 32 GB/CHAT BOTS LOCAL/huggingface_cache'

## LOCAL
# llm = HuggingFacePipeline.from_model_id(
#     model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
#     task='text-generation',
#     pipeline_kwargs=dict(
#         do_sample=True,
#         temperature=0.5,
#         max_new_tokens=300
#     )
# )


prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

model = ChatHuggingFace(llm=llm)
# model = ChatOpenAI()

parser = StrOutputParser()
chain = prompt | model
# result = chain.invoke({'topic':'cricket'})
result = invoke_with_retry(chain, {'topic':'cricket'})

# print(result)
# chain.get_graph().print_ascii()
if result:
    print(result)
    chain.get_graph().print_ascii()
