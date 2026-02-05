## API
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
load_dotenv()
from huggingface_hub.utils import HfHubHTTPError
import time

## API ~ TinyLlama/TinyLlama-1.1B-Chat-v1.0 ~ google/gemma-2-2b-it
llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)

## API
# Retry wrapper
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
    print("❌ Failed to get a valid response after retries.")
    return None


## LOCAL
# from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
# from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
# from langchain_core.prompts import PromptTemplate
# from langchain.output_parsers import StructuredOutputParser, ResponseSchema
# import os
# os.environ['HF_HOME'] = 'D:/PENDRIVE 32 GB/CHAT BOTS LOCAL/huggingface_cache'

# ## LOCAL
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
schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give 3 fact about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser

# result = chain.invoke({'topic':'black hole'})
# print(result)
result = invoke_with_retry(chain, {'topic': 'black hole'})
if result:
    print(result)
