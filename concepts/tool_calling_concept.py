# Setup
from helper import get_openai_api_key

OPENAI_API_KEY = get_openai_api_key()

# Jupiter runs an event loop behind the scenes and a lot of our modules use async
# and to make async play nice with Jupiter notebooks, we need to import this.
import nest_asyncio

nest_asyncio.apply()

from llama_index.core.tools import FunctionTool

# The core abstraction in LlamaIndex is the class FunctionTool
# This function tool wraps any given Python function that you feed it
# 
# class FunctionTool(AsyncBaseTool):
#     """Function Tool.
#     A tool that takes in a function.
#     """

# The functions have type annotations for both x and y variables, as well
# as the docstring. This is not just for stylistic purposes!
# It's important to declare the functions like that because these things
# will be used as a prompt for the LLM.

def add(x: int, y: int) -> int:
    """Adds two integers together."""
    return x + y

def mystery(x: int, y: int) -> int: 
    """Mystery function that operates on top of two numbers."""
    return (x + y) * (x + y)

add_tool = FunctionTool.from_defaults(fn=add)
mystery_tool = FunctionTool.from_defaults(fn=mystery)

# Setting gpt-3.5-turbo as LLM of this program
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")
# The product_and_call functions takes in a set of tools, as well as an
# input prompt string or series of chat messages.
# And the it's able to both make a decision of the tool to call, as well
# as call the tool itself and get back the final response
response = llm.predict_and_call(
    [add_tool, mystery_tool], 
    "Tell me the output of the mystery function on 2 and 9", 
    verbose=True
)
print(str(response))

# The LLM will call the right tools and also infer the rigth parameters!