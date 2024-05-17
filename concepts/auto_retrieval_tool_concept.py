# Setup
from helper import get_openai_api_key

OPENAI_API_KEY = get_openai_api_key()

# Jupiter runs an event loop behind the scenes and a lot of our modules use async
# and to make async play nice with Jupiter notebooks, we need to import this.
import nest_asyncio

nest_asyncio.apply()

from llama_index.core.tools import FunctionTool

from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader(input_files=["metagpt.pdf"]).load_data()

# Splitting documents in nodes 

from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(chunk_size=1024)

nodes = splitter.get_nodes_from_documents(documents)

# Print first node and metadata
print(nodes[0].get_content(metadata_mode="all"))

# Build a vector index from nodes
from llama_index.core import VectorStoreIndex

vector_index = VectorStoreIndex(nodes)
query_engine = vector_index.as_query_engine(similarity_top_k=2)

# Define metadata filter (arguments)
from llama_index.core.vector_stores import MetadataFilters

query_engine = vector_index.as_query_engine(
    similarity_top_k=2,
    filters=MetadataFilters.from_dicts(
        [
            {"key": "page_label", "value": "2"}
        ]
    )
)

response = query_engine.query(
    "What are some high-level results of MetaGPT?", 
)

print(str(response))

# Printing the metadata attached to these source nodes
for n in response.source_nodes:
    print(n.metadata)