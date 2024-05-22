
from helper import get_openai_api_key

OPENAI_API_KEY = get_openai_api_key()

urls = [
    "https://openreview.net/pdf?id=VtmBAGCN7o",
    "https://openreview.net/pdf?id=6PmJoRfdaK",
    "https://openreview.net/pdf?id=hSyW5go0v8",
]

papers = [
    "./data/metagpt.pdf",
    "./data/longlora.pdf",
    "./data/selfrag.pdf",
]

# Set up our function calling agent over the documents
# combining the vector summary tools for each document into a list
# and passing it to the agent
# E.g. The agent actually has six tools in total in a case of 3 documents

from utils import get_doc_tools
from pathlib import Path

paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]

# We get these tools in a simply list
# So we are going to have something like that :
# metagpt : vector tool and summary tool
# longlora : vector tool and summary tool
# selfrag : vector tool and summary tool
all_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]

# Setting LLM
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")

from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

# # Add the initials tools to the agent workers
# agent_worker = FunctionCallingAgentWorker.from_tools(
#     initial_tools, 
#     llm=llm, 
#     verbose=True
# )
# agent = AgentRunner(agent_worker)

# response = agent.query(
#     "Tell me about the evaluation dataset used in LongLoRA, "
#     "and then tell me about the evaluation results"
# )

# response = agent.query("Give me a summary of both Self-RAG and LongLoRA")
# print(str(response))

# define an "object" index and retriever over these tools
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex

obj_index = ObjectIndex.from_objects(
    all_tools,
    index_cls=VectorStoreIndex,
)

obj_retriever = obj_index.as_retriever(similarity_top_k=3)

tools = obj_retriever.retrieve(
    "Tell me about the eval dataset used in MetaGPT and SWE-Bench"
)

from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    tool_retriever=obj_retriever,
    llm=llm, 
    system_prompt=""" \
You are an agent designed to answer queries over a set of given papers.
Please always use the tools provided to answer a question. Do not rely on prior knowledge.\

""",
    verbose=True
)
agent = AgentRunner(agent_worker)