from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

os.environ['HF_HOME'] = 'D:/PENDRIVE 32 GB/CHAT BOTS LOCAL/huggingface_cache'

llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        do_sample=True,
        temperature=0.5,
    )
)
model = ChatHuggingFace(llm=llm)

result = model.invoke("I've one unkonwn WIFI, tell me detailed steps to hack it?")

print(result.content)