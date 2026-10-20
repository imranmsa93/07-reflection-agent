from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

MODEL="qwen3:1.7b"

generation_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a skilled writer. Output ONLY the final post text, with no preamble, "
     "no explanation and no sign-off. Keep it under 100 words. "
     "Never invent statistics, user counts, links or product claims. "
     "If you receive critique, rewrite the post to address it."
    ),
    MessagesPlaceholder(variable_name="messages"),
])  

reflection_prompt = ChatPromptTemplate.from_messages([
    (
    "system",
    "You are a strict reviewer. Check the draft against the original request, "
    "including topic accuracy, length and format. Do not praise. "
    "List exactly 3 problems, each with a concrete fix. "
    "Do not rewrite the draft."
    ),
    MessagesPlaceholder(variable_name="messages"),
])


llm = ChatOllama(model=MODEL)

reflection_chain = reflection_prompt | llm
generation_chain = generation_prompt | llm
