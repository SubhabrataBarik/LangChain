## API
# from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from dotenv import load_dotenv
# load_dotenv()
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
# from langchain_core.output_parsers import PydanticOutputParser
# from pydantic import BaseModel, Field
# from typing import Literal
# from huggingface_hub.utils import HfHubHTTPError
# import time

## API ~ TinyLlama/TinyLlama-1.1B-Chat-v1.0 ~ google/gemma-2-2b-it
# llm = HuggingFaceEndpoint(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation"
# )

## API
# def invoke_with_retry(chain, input_dict, retries=3, delay=5):
#     for attempt in range(retries):
#         try:
#             return chain.invoke(input_dict)
#         except HfHubHTTPError as e:
#             print(f"[Attempt {attempt + 1}] HF Hub Error: {e}")
#             time.sleep(delay)
#         except Exception as e:
#             print(f"[Attempt {attempt + 1}] Error: {e}")
#             time.sleep(delay)
#     print("❌ Failed after maximum retry attempts.")
#     return None


## LOCAL
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
import os
os.environ['HF_HOME'] = 'D:/PENDRIVE 32 GB/CHAT BOTS LOCAL/huggingface_cache'

## LOCAL
llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        do_sample=True,
        temperature=0.5,
        max_new_tokens=300
    )
)


# model = ChatOpenAI()
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x:x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain
feedback_input = {'feedback': 'This is a beautiful phone'}

feedback_input = chain.invoke(feedback_input)
# final_output = invoke_with_retry(chain, feedback_input)

if final_output:
    print("Generated response:")
    print(final_output)
    chain.get_graph().print_ascii()