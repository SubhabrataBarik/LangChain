## API
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
load_dotenv()
from huggingface_hub.utils import HfHubHTTPError
import time

## API ~ TinyLlama/TinyLlama-1.1B-Chat-v1.0 ~ google/gemma-2-2b-it
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
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
            print(f"[Attempt {attempt + 1}] Error: {e}")
            time.sleep(delay)
    print("❌ Failed after maximum retry attempts.")
    return None


## LOCAL
# from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
# from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import PydanticOutputParser
# from pydantic import BaseModel, Field
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


model = ChatHuggingFace(llm=llm)
class Person(BaseModel):

    name: str = Field(description='Name of the person')
    age: int = Field(gt=18, description='Age of the person')
    city: str = Field(description='Name of the city the person belongs to')

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate the name, age and city of a fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser

# final_result = chain.invoke({'place':'sri lankan'})
# print(final_result)
final_result = invoke_with_retry(chain, {'place': 'Sri Lankan'})
if final_result:
    print(final_result)