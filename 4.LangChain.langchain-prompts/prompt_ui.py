## PAPER SUMMERY WEBSITE ~ list of msg ~ DYNAMIC PROMPT

## LOCAL
# from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
# from langchain_core.prompts import PromptTemplate,load_prompt
# import streamlit as st
# import os
# os.environ['HF_HOME'] = 'D:/PENDRIVE 32 GB/CHAT BOTS LOCAL/huggingface_cache'

## LOCAL
# llm = HuggingFacePipeline.from_model_id(
#     model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
#     task='text-generation',
#     pipeline_kwargs=dict(
#         temperature=0.5,
#         max_new_tokens=100
#     )
# )


## API
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate,load_prompt
from huggingface_hub.utils import HfHubHTTPError
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import time

## API
llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)


model = ChatHuggingFace(llm=llm)

st.header('Reasearch Tool')

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

template = load_prompt('template.json')


## API
# Retry wrapper function
def invoke_with_retry(chain, inputs, retries=3, delay=5):
    for attempt in range(retries):
        try:
            return chain.invoke(inputs)
        except HfHubHTTPError as e:
            # st.warning(f"[Attempt {attempt+1}] Hugging Face Error: {e}")
            time.sleep(delay)
        except Exception as e:
            # st.warning(f"[Attempt {attempt+1}] Other Error: {e}")
            time.sleep(delay)
    st.error("⚠️ Unable to generate summary after multiple attempts.")
    return None

## API
if st.button('🧠 Summarize'):
    with st.spinner('Generating summary...'):
        chain = template | model
        result = invoke_with_retry(chain, {
            'paper_input': paper_input,
            'style_input': style_input,
            'length_input': length_input
        })
        if result:
            st.subheader("📝 Summary:")
            st.write(result.content)

## LOCAL
# if st.button('🧠 Summarize'):
#     with st.spinner('Generating summary...'):
#         chain = template | model
#         result = chain.invoke({
#             'paper_input':paper_input,
#             'style_input':style_input,
#             'length_input':length_input
#         })
#         st.write(result.content)