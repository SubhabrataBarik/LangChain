from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from transformers import AutoModelForCausalLM

from huggingface_hub.utils import HfHubHTTPError
from dotenv import load_dotenv
import time

load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation"
# )
# model = ChatHuggingFace(llm=llm)

model = AutoModelForCausalLM.from_pretrained('JetBrains/Mellum-4b-base')

## Retry function
# def invoke_with_retry(model, user_input, retries=3, delay=5):
#     for attempt in range(retries):
#         try:
#             return model.invoke(user_input)
#         except HfHubHTTPError as e:
#             # print(f"[Attempt {attempt+1}] Hugging Face Error: {e}")
#             time.sleep(delay)
#         except Exception as e:
#             # print(f"[Attempt {attempt+1}] Other Error: {e}")
#             time.sleep(delay)
#     print("❌ Unable to get a response after multiple attempts.")
#     return None
# user_input=input("You: ")
# result = invoke_with_retry(model, user_input)

result = model.invoke("What is the capital of India")
if result:
    print(result.content)