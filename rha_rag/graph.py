"""LangGraph construction — nodes, edges, and build function."""

from pathlib import Path
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════
# Prompts (loaded from prompts/*.txt)
# ═══════════════════════════════════════════════════════════════

_PROMPT_DIR = Path(__file__).parent.parent / "prompts"

def _load_prompt(name: str) -> str:
    return (_PROMPT_DIR / name).read_text(encoding="utf-8")

CLARIFY_PROMPT = _load_prompt("clarify.txt")
GRADE_PROMPT   = _load_prompt("grade.txt")
REASON_PROMPT  = _load_prompt("reason.txt")
VERIFY_PROMPT  = _load_prompt("verify.txt")
GENERATE_PROMPT = _load_prompt("generate.txt")


class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Relevance score: 'yes' or 'no'")


# ═══════════════════════════════════════════════════════════════
# Node factories — accept models as closure arguments
# ═══════════════════════════════════════════════════════════════

def make_clarify_question(response_model):
    def clarify_question(state: MessagesState):
        question = state["messages"][0].content
        prompt = CLARIFY_PROMPT.format(question=question)
        response = response_model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [HumanMessage(content=f"CLARIFIED: {response.content}")]}
    return clarify_question


def make_generate_query_or_respond(response_model, retriever_tool):
    def generate_query_or_respond(state: MessagesState):
        response = response_model.bind_tools([retriever_tool]).invoke(state["messages"])
        return {"messages": [response]}
    return generate_query_or_respond


def make_grade_documents(grader_model):
    def grade_documents(state: MessagesState):
        question = state["messages"][0].content
        context = state["messages"][-1].content
        prompt = GRADE_PROMPT.format(question=question, context=context)
        response = grader_model.with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": prompt}]
        )
        return {"messages": [HumanMessage(
            content=f"DOCUMENT GRADE: {response.binary_score}\n\nRetrieved context:\n{context}"
        )]}
    return grade_documents


def make_reason(response_model):
    def reason(state: MessagesState):
        clarified = ""
        for msg in reversed(state["messages"]):
            content = getattr(msg, "content", "")
            if content.startswith("CLARIFIED:"):
                clarified = content
                break
        if not clarified:
            clarified = state["messages"][0].content
        context = state["messages"][-1].content
        prompt = REASON_PROMPT.format(clarified=clarified, context=context)
        response = response_model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [HumanMessage(content=f"REASONING: {response.content}")]}
    return reason


def make_verify(response_model):
    def verify(state: MessagesState):
        clarified = ""
        reasoning_text = ""
        for msg in reversed(state["messages"]):
            content = getattr(msg, "content", "")
            if content.startswith("REASONING:") and not reasoning_text:
                reasoning_text = content
            elif content.startswith("CLARIFIED:") and not clarified:
                clarified = content
        if not clarified:
            clarified = state["messages"][0].content
        prompt = VERIFY_PROMPT.format(clarified=clarified, reasoning=reasoning_text)
        response = response_model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [HumanMessage(content=f"VERIFIED: {response.content}")]}
    return verify


def make_generate_answer(response_model):
    def generate_answer(state: MessagesState):
        question = state["messages"][0].content
        context = ""
        verified = ""
        for msg in reversed(state["messages"]):
            content = getattr(msg, "content", "")
            if content.startswith("VERIFIED:") and not verified:
                verified = content
            if getattr(msg, "type", "") == "tool" and not context:
                context = content
        if not context:
            context = state["messages"][-1].content
        prompt = GENERATE_PROMPT.format(
            question=question, context=context, verified=verified
        )
        response = response_model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [response]}
    return generate_answer


# ═══════════════════════════════════════════════════════════════
# Graph assembly
# ═══════════════════════════════════════════════════════════════

from langchain.tools import tool


def make_retriever_tool(retriever):
    """Create a LangChain tool wrapping the retriever."""

    @tool
    def retrieve_content(query: str) -> str:
        """Search and return information from the local knowledge base."""
        if retriever is None:
            return "No documents available in the knowledge base."
        results = retriever.invoke(query)
        if not results:
            return "No relevant documents found."
        return "\n\n".join(
            f"[Source: {d.metadata.get('source', 'unknown')}]: {d.page_content}"
            for d in results
        )

    return retrieve_content


def build_graph(retriever, response_model, grader_model):
    """Build and return the compiled LangGraph StateGraph.

    Graph flow:
        START -> clarify -> generate_query -> [conditional] -> retrieve
                      |                                      |
                      |                               grade -> reason -> verify -> generate_answer -> END
                      +-> END (no tool call)
    """
    from langgraph.graph import END, START, StateGraph
    from langgraph.prebuilt import ToolNode

    retriever_tool = make_retriever_tool(retriever)

    workflow = StateGraph(MessagesState)

    # Create nodes with model references
    workflow.add_node("clarify", make_clarify_question(response_model))
    workflow.add_node("generate_query", make_generate_query_or_respond(response_model, retriever_tool))
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("grade", make_grade_documents(grader_model))
    workflow.add_node("reason", make_reason(response_model))
    workflow.add_node("verify", make_verify(response_model))
    workflow.add_node("generate_answer", make_generate_answer(response_model))

    # Entry
    workflow.add_edge(START, "clarify")
    workflow.add_edge("clarify", "generate_query")

    # Conditional: tool call or direct answer
    def route_on_tool_calls(state: MessagesState):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    workflow.add_conditional_edges(
        "generate_query",
        route_on_tool_calls,
        {"tools": "retrieve", END: END},
    )

    # Linear pipeline after retrieval
    workflow.add_edge("retrieve", "grade")
    workflow.add_edge("grade", "reason")
    workflow.add_edge("reason", "verify")
    workflow.add_edge("verify", "generate_answer")
    workflow.add_edge("generate_answer", END)

    return workflow.compile()
