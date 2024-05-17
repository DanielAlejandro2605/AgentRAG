# Integrating Metadata Filters into a retrieval tool
# Enable more precise retrieval by accepting a query string
# and optional metada filters, such as page numbers

# The LLM can intelligently infer relevant metadata filter (e.g, page numbers)
# based on the user's query.

# Note :    The metadata is not only limited to page numbers
#           You can define whatever metadata you want like:
#               - Sections IDs
#               - Headers, footers
#               - Anything else through LlamaIndex Abstractions

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

from typing import List
from llama_index.core.vector_stores import FilterCondition

def vector_query(
    query: str, 
    page_numbers: List[str]
) -> str:
    """Perform a vector search over an index.
    
    query (str): the string query to be embedded.
    page_numbers (List[str]): Filter by set of pages. Leave BLANK if we want to perform a vector search
        over all pages. Otherwise, filter by the set of specified pages.
    
    """

    metadata_dicts = [
        {"key": "page_label", "value": p} for p in page_numbers
    ]
    
    query_engine = vector_index.as_query_engine(
        similarity_top_k=2,
        filters=MetadataFilters.from_dicts(
            metadata_dicts,
            condition=FilterCondition.OR
        )
    )
    response = query_engine.query(query)
    return response
    

vector_query_tool = FunctionTool.from_defaults(
    name="vector_tool",
    fn=vector_query
)

# Let's add some other tools!

from llama_index.core import SummaryIndex
from llama_index.core.tools import QueryEngineTool

summary_index = SummaryIndex(nodes)
summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)
summary_tool = QueryEngineTool.from_defaults(
    name="summary_tool",
    query_engine=summary_query_engine,
    description=(
        "Useful if you want to get a summary of MetaGPT"
    ),
)

response = llm.predict_and_call(
    [vector_query_tool, summary_tool], 
    "What are the MetaGPT comparisons with ChatDev described on page 8?", 
    verbose=True
)

for n in response.source_nodes:
    print(n.metadata)

response = llm.predict_and_call(
    [vector_query_tool, summary_tool], 
    "What is a summary of the paper?", 
    verbose=True
)

print(str(response))