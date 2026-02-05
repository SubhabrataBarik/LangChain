from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from huggingface_hub.utils import HfHubHTTPError
import time


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)



## Retry wrapper function
def invoke_with_retry(model, prompt, retries=3, delay=5):
    for attempt in range(retries):
        try:
            return model.invoke(prompt)
        except HfHubHTTPError as e:
            # print(f"[Attempt {attempt + 1}] HF Error: {e}")
            time.sleep(delay)
        except Exception as e:
            # print(f"[Attempt {attempt + 1}] Other Error: {e}")
            time.sleep(delay)
    print("❌ Could not get a response after multiple attempts.")
    return None


# 1st prompt -> detailed report
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. /n {text}',
    input_variables=['text']
)

prompt1 = template1.invoke({'topic':'black hole'})
# result = model.invoke(prompt1)
result = invoke_with_retry(model, prompt1)

# prompt2 = template2.invoke({'text':result.content})
# result1 = model.invoke(prompt2)
# print(result1.content)
if result:
    # Invoke second template for summary
    prompt2 = template2.invoke({'text': result.content})
    result1 = invoke_with_retry(model, prompt2)

    if result1:
        print(result1.content)