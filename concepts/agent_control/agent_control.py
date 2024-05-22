from helper import get_openai_api_key

OPENAI_API_KEY = get_openai_api_key()

# Get vector query and summary query tools from a document.
from utils import get_doc_tools

vector_tool, summary_tool = get_doc_tools("metagpt.pdf", "metagpt")

# Set LLM model 
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo", temperature=0)

# Defining AgentWorker and AgentRunner
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

# class AgentRunner(BaseAgentRunner):
#     """Agent runner.

#     Top-level agent orchestrator that can create tasks, run each step in a task,
#     or run a task e2e. Stores state and keeps track of tasks.

#     Args:
#         agent_worker (BaseAgentWorker): step executor
#         chat_history (Optional[List[ChatMessage]], optional): chat history. Defaults to None.
#         state (Optional[AgentState], optional): agent state. Defaults to None.
#         memory (Optional[BaseMemory], optional): memory. Defaults to None.
#         llm (Optional[LLM], optional): LLM. Defaults to None.
#         callback_manager (Optional[CallbackManager], optional): callback manager. Defaults to None.
#         init_task_state_kwargs (Optional[dict], optional): init task state kwargs. Defaults to None.

#     """

agent_worker = FunctionCallingAgentWorker.from_tools(
    [vector_tool, summary_tool], 
    llm=llm, 
    verbose=True
)
agent = AgentRunner(agent_worker)

# Use low-level API 
# Creating a task object from the user query

task = agent.create_task(
    "Tell me about the agent roles in MetaGPT, "
    "and then how they communicate with each other."
)

# Executing a single step of this task
step_output = agent.run_step(task.task_id)

# Expecting logs and output of the agent
completed_steps = agent.get_completed_steps(task.task_id)
print(f"Num completed for task {task.task_id}: {len(completed_steps)}")
print(completed_steps[0].output.sources[0].raw_output)

# Look at any upcming steps for the agent
upcoming_steps = agent.get_upcoming_steps(task.task_id)
print(f"Num upcoming steps for task {task.task_id}: {len(upcoming_steps)}")
upcoming_steps[0]

# Insert more information between step
step_output = agent.run_step(
    task.task_id, input="What about how agents share information?"
)

# Is the last step in the queue of steps? 
step_output = agent.run_step(task.task_id)
print(step_output.is_last)

# Getting the final response
response = agent.finalize_response(task.task_id)
print(str(response))