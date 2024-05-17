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
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002", timeout=60)

# Setting vector and summary index from nodes
from llama_index.core import SummaryIndex, VectorStoreIndex

# class SummaryIndex(BaseIndex[IndexList]):
#     """Summary Index.

#     The summary index is a simple data structure where nodes are stored in
#     a sequence. During index construction, the document texts are
#     chunked up, converted to nodes, and stored in a list."""

#     During query time, the summary index iterates through the nodes
#     with some optional filter parameters, and synthesizes an
#     answer from all the nodes.

summary_index = SummaryIndex(nodes)

# class VectorStoreIndex(BaseIndex[IndexDict]):
#     """Vector Store Index.

#     Args:
#         use_async (bool): Whether to use asynchronous calls. Defaults to False.
#         show_progress (bool): Whether to show tqdm progress bars. Defaults to False.
#         store_nodes_override (bool): set to True to always store Node objects in index
#             store and document store even if vector store keeps text. Defaults to False
#     """

vector_index = VectorStoreIndex(nodes)

# We can use this index as query engine for the router
# using the function inherited from the BaseIndex class `as_query_engine`
# 
# def as_query_engine(
#     self, llm: Optional[LLMType] = None, **kwargs: Any
#     ) -> BaseQueryEngine:
#     """Convert the index to a query engine.

#     Calls `index.as_retriever(**kwargs)` to get the retriever and then wraps it in a
#     `RetrieverQueryEngine.from_args(retriever, **kwrags)` call.
#     """ 

summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True
)

vector_query_engine = vector_index.as_query_engine()


# We declare one query tool for each query engine
# A query tool now is just the query engine with metadata, 
# specificaly a description of what types of questions the toll can answer. 

from llama_index.core.tools import QueryEngineTool

# class QueryEngineTool(AsyncBaseTool):
#     """Query engine tool.

#     A tool making use of a query engine.

#     Args:
#         query_engine (BaseQueryEngine): A query engine.
#         metadata (ToolMetadata): The associated metadata of the query engine.
#     """

summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    description=(
        "Useful for summarization questions related to MetaGPT"
    ),
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description=(
        "Useful for retrieving specific context from the MetaGPT paper."
    ),
)

# Defining router and selector for this router adding both query tools

from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

# class RouterQueryEngine(BaseQueryEngine):
#     """Router query engine.

#     Selects one out of several candidate query engines to execute a query.

#     Args:
#         selector (BaseSelector): A selector that chooses one out of many options based
#             on each candidate's metadata and query.
#         query_engine_tools (Sequence[QueryEngineTool]): A sequence of candidate
#             query engines. They must be wrapped as tools to expose metadata to
#             the selector.
#         service_context (Optional[ServiceContext]): A service context.
#         summarizer (Optional[TreeSummarize]): Tree summarizer to summarize sub-results.

#     """

# class LLMSingleSelector(BaseSelector):
#     """LLM single selector.

#     LLM-based selector that chooses one out of many options.

#     Args:
#         LLM (LLM): An LLM.
#         prompt (SingleSelectPrompt): A LLM prompt for selecting one out of many options.
#     """


query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        summary_tool,
        vector_tool,
    ],
    verbose=True
)

response = query_engine.query("What is the summary of the document?")
print(str(response))

response = query_engine.query(
    "How do agents share information with other agents?"
)

print(str(response))