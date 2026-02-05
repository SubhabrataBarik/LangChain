## API
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from huggingface_hub.utils import HfHubHTTPError
import time
from dotenv import load_dotenv
load_dotenv()
from langchain.text_splitter import CharacterTextSplitter

## API
llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    max_new_tokens=500,
    temperature=0.7
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


from langchain_community.document_loaders import WebBaseLoader
# from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

model = ChatHuggingFace(llm=llm)
# model = ChatOpenAI()

prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question','text']
)

parser = StrOutputParser()

url = 'https://www.flipkart.com/apple-macbook-air-m2-16-gb-256-gb-ssd-macos-sequoia-mc7x4hn-a/p/itmdc5308fa78421'
loader = WebBaseLoader(url)

docs = loader.load()

# ---- Split Long Text ----
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1500,
    chunk_overlap=100,
    length_function=len
)
chunks = text_splitter.split_text(docs[0].page_content)
chain = prompt | model | parser

# print(chain.invoke({'question':'What is the prodcut that we are talking about?', 'text':docs[0].page_content}))

# ---- Ask Question on Each Chunk Until a Confident Answer Found ----
question = "What is the product that we are talking about?"
for i, chunk in enumerate(chunks):
    print(f"\n🔍 Chunk {i+1}:")
    response = invoke_with_retry(chain, {'question': question, 'text': chunk})
    print(response)
    if response and any(keyword in response.lower() for keyword in ['macbook', 'laptop', 'apple']):
        print("\n✅ Relevant answer found. Stopping further processing.")
        break