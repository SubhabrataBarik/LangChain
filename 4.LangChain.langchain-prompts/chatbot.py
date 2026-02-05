## CHATBOT with list of msg ~ Static history

## LOCAL
# from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
# from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
# import os
# os.environ['HF_HOME'] = 'D:/PENDRIVE 32 GB/CHAT BOTS LOCAL/huggingface_cache'

## API
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from huggingface_hub.utils import HfHubHTTPError
from dotenv import load_dotenv
load_dotenv()
import time


## API
llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)

## LOCAL
# llm = HuggingFacePipeline.from_model_id(
#     model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
#     task='text-generation',
#     pipeline_kwargs=dict(
#         do_sample=True,
#         temperature=0.5,
#         max_new_tokens=15
#     )
# )

model = ChatHuggingFace(llm=llm)

chat_history = [
    SystemMessage(content='You are a helpful AI assistant')
]

## API
def invoke_with_retry(model, chat_history, retries=3, delay=5):
    for attempt in range(retries):
        try:
            return model.invoke(chat_history)
        except HfHubHTTPError as e:
            # print(f"[Attempt {attempt+1}] Error: {e}")
            # print(f"Retrying in {delay} seconds...")
            time.sleep(delay)
        except Exception as e:
            # print(f"[Attempt {attempt+1}] Unexpected error: {e}")
            # print(f"Retrying in {delay} seconds...")
            time.sleep(delay)
    raise RuntimeError("Model failed to respond after multiple attempts.")


while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() == 'exit':
        break
    
    try:
        result = invoke_with_retry(model, chat_history)
        chat_history.append(AIMessage(content=result.content))
        print("AI:", result.content)
    except RuntimeError as e:
        print("AI: Sorry, I'm currently unavailable. Please try again later.")
        break


print(chat_history)