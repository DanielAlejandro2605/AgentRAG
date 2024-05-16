# Setup
from helper import get_openai_api_key

OPENAI_API_KEY = get_openai_api_key()

# Jupiter runs an event loop behind the scenes and a lot of our modules use async
# and to make async play nice with Jupiter notebooks, we need to import this.
import nest_asyncio

nest_asyncio.apply()

from llama_index.core import SimpleDirectoryReader

# The SimpleDirectoryReader Class 
# Load files from file directory.
# Automatically select the best file reader given file extensions.
#   Args:
#       input_dir (str): Path to the directory.
#       input_files (List): List of file paths to read

documents = SimpleDirectoryReader(input_files=["metagpt.pdf"]).load_data()

# Splitting documents in nodes 

from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(chunk_size=1024)

nodes = splitter.get_nodes_from_documents(documents)

# Setting globals models gpt-3.5-turbo and text-embedding-ada-002 
# for 
# GPT-3.5 Turbo :   GPT-3.5 Turbo models can understand and generate natural language or code
# text-embedding-ada-002 :  Embeddings are a numerical representation of text that can be used to measure 
#                           the relatedness between two pieces of text.

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")