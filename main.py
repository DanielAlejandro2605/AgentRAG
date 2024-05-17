from utils import get_router_query_engine

query_engine = get_router_query_engine("metagpt.pdf")

response = query_engine.query("Tell me about the ablation study results?")
print(str(response))