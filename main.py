from utils.router import get_router_query_engine

# Setup
from utils.helper import get_openai_api_key

OPENAI_API_KEY = get_openai_api_key()

query_engine = get_router_query_engine("./data/metagpt.pdf")

response = query_engine.query("Tell me about the ablation study results?")
print(str(response))