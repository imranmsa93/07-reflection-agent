from dotenv import load_dotenv
import os
load_dotenv()
from chains import reflection_chain,generation_chain
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tool_node
from typing import TypedDict,Annotated 
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import MessagesState, add_messages

def generate_node(state:MessagesState):
    reponse = generation_chain.invoke({"messages":state["messages"]})
    return {"messages":[reponse]}

def reflection_node(state:MessagesState):
    request = state["messages"][0].content
    draft = state["messages"][-1].content
    prompt_text = f"Original request:\n{request}\n\nDraft:\n{draft}"
    reponse = reflection_chain.invoke({"messages":[HumanMessage(content=prompt_text)]})
    return {"messages":[HumanMessage(content =reponse.content)]}

def should_continue(state:MessagesState):
    if len(state["messages"]) >= 4:
        return END
    return 'reflection'

builder = StateGraph(MessagesState)

builder.add_node('generate',generate_node)
builder.add_node('reflection',reflection_node)

builder.add_edge(START,'generate')
builder.add_conditional_edges('generate', should_continue,['reflection', END])
builder.add_edge('reflection','generate')

app = builder.compile()


def main():
    #print(app.get_graph().draw_mermaid())    # text, no internet
    #app.get_graph().draw_mermaid_png(output_file_path="flow.png")
    print("Hello from 07-reflection-agent!")
    res = app.invoke(
        {"messages": [HumanMessage(content= "Write a short LinkedIn post about LangGraph. "
                        "Facts you may use: LangGraph is a framework for building stateful, "
                        "graph-based agent workflows. It uses nodes and edges, supports loops and "
                        "conditional routing, and can persist memory between runs. "
                        "Do not state anything beyond these facts."
                    )]}
    )
    for m in res["messages"]:
        m.pretty_print()


if __name__ == "__main__":
    main()
