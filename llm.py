from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
load_dotenv()

llm = ChatOpenAI(model="gpt-4", temperature=0)