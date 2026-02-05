## API
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from huggingface_hub.utils import HfHubHTTPError
import time
from dotenv import load_dotenv
load_dotenv()

## API
llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
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


## LOCAL
# from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
# import os
# os.environ['HF_HOME'] = 'D:/PENDRIVE 32 GB/CHAT BOTS LOCAL/huggingface_cache'

## LOCAL
# llm = HuggingFacePipeline.from_model_id(
#     model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
#     task='text-generation',
#     pipeline_kwargs=dict(
#         temperature=1,
#         max_new_tokens=100
#     )
# )


from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

model = ChatHuggingFace(llm=llm)
# model = ChatOpenAI()

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

# print(chain.invoke({'topic':'AI'}))

result = invoke_with_retry(chain, {'topic':'AI'})
print("\n🃏 Joke Explanation:")
print(result)

