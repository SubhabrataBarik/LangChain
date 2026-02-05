## API
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

## API
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)


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
#         max_new_tokens=200
#     )
# )


model = ChatHuggingFace(llm=llm)
# model = ChatOpenAI()

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

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':'black hole'})

print(result)