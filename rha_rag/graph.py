"""LangGraph construction — nodes, edges, and build function."""

from pathlib import Path
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field


class RhaState(MessagesState):
    """Pipeline state.

    `messages` carries the conversation: prior Q&A (prepended by the server for
    multi-turn memory) followed by the current turn's messages. `question` is
    the current turn's question (used by nodes that previously read
    `messages[0]`, which no longer holds once history is prepended). `history`
    is a condensed prior-Q&A string injected into the clarify/generate prompts.
    `fast_mode` skips clarify / grade / reason / verify — retrieve then answer.
    `visual_spec` carries the chart-spec JSON produced by the visualize node,
    forwarded to the frontend via the SSE channel.
    """

    question: str
    history: str
    fast_mode: bool = False
    visual_spec: str = ""


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
    def clarify_question(state: RhaState):
        question = state["question"]
        prompt = CLARIFY_PROMPT.format(question=question, history=state.get("history", ""))
        response = response_model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [HumanMessage(content=f"CLARIFIED: {response.content}")]}
    return clarify_question


def make_generate_query_or_respond(response_model, retriever_tool):
    def generate_query_or_respond(state: RhaState):
        response = response_model.bind_tools([retriever_tool]).invoke(state["messages"])
        return {"messages": [response]}
    return generate_query_or_respond


def make_grade_documents(grader_model):
    def grade_documents(state: RhaState):
        question = state["question"]
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
    def reason(state: RhaState):
        clarified = ""
        for msg in reversed(state["messages"]):
            content = getattr(msg, "content", "")
            if content.startswith("CLARIFIED:"):
                clarified = content
                break
        if not clarified:
            clarified = state["question"]
        context = state["messages"][-1].content
        prompt = REASON_PROMPT.format(clarified=clarified, context=context)
        response = response_model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [HumanMessage(content=f"REASONING: {response.content}")]}
    return reason


def make_verify(response_model):
    def verify(state: RhaState):
        clarified = ""
        reasoning_text = ""
        for msg in reversed(state["messages"]):
            content = getattr(msg, "content", "")
            if content.startswith("REASONING:") and not reasoning_text:
                reasoning_text = content
            elif content.startswith("CLARIFIED:") and not clarified:
                clarified = content
        if not clarified:
            clarified = state["question"]
        prompt = VERIFY_PROMPT.format(clarified=clarified, reasoning=reasoning_text)
        response = response_model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [HumanMessage(content=f"VERIFIED: {response.content}")]}
    return verify


def make_generate_answer(response_model):
    def generate_answer(state: RhaState):
        question = state["question"]
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
            question=question,
            context=context,
            verified=verified,
            history=state.get("history", ""),
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

    Full path (fast_mode=False):
        START -> clarify -> generate_query -> [conditional] -> retrieve
                           -> grade -> reason -> verify -> generate_answer
                           -> visualize -> END

    Fast path (fast_mode=True):
        START -> generate_query -> [conditional] -> retrieve
                                 -> generate_answer -> visualize -> END

    The visualize node runs in both modes. It reads the prior
    generate_answer output + retrieved context, asks the LLM for a JSON
    ChartSpec, validates with Pydantic, and emits via SSE.
    """
    from langgraph.graph import END, START, StateGraph
    from langgraph.prebuilt import ToolNode

    retriever_tool = make_retriever_tool(retriever)

    workflow = StateGraph(RhaState)

    # Nodes
    workflow.add_node("clarify", make_clarify_question(response_model))
    workflow.add_node("generate_query", make_generate_query_or_respond(response_model, retriever_tool))
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("grade", make_grade_documents(grader_model))
    workflow.add_node("reason", make_reason(response_model))
    workflow.add_node("verify", make_verify(response_model))
    workflow.add_node("generate_answer", make_generate_answer(response_model))

    # Visualization node — runs in BOTH fast and full modes. Reads the
    # generate_answer output + retrieved context, asks the LLM for a
    # JSON ChartSpec, validates with Pydantic, emits via SSE.
    from rha_rag.viz import make_visualize
    workflow.add_node("visualize", make_visualize(response_model))

    # Entry: skip clarify in fast mode.
    def _route_entry(state: RhaState) -> str:
        return "generate_query" if state.get("fast_mode") else "clarify"

    workflow.add_conditional_edges(
        START,
        _route_entry,
        {"clarify": "clarify", "generate_query": "generate_query"},
    )
    workflow.add_edge("clarify", "generate_query")

    # Tool-call gate.
    def _route_tools(state: RhaState) -> str:
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END

    workflow.add_conditional_edges(
        "generate_query",
        _route_tools,
        {"tools": "retrieve", END: END},
    )

    # After retrieval: skip the reasoning nodes in fast mode.
    def _route_after_retrieve(state: RhaState) -> str:
        return "generate_answer" if state.get("fast_mode") else "grade"

    workflow.add_conditional_edges(
        "retrieve",
        _route_after_retrieve,
        {"generate_answer": "generate_answer", "grade": "grade"},
    )

    # Full-path chain (only reached when fast_mode is False).
    workflow.add_edge("grade", "reason")
    workflow.add_edge("reason", "verify")
    workflow.add_edge("verify", "generate_answer")
    # Visualization runs after the answer in BOTH modes — chart spec
    # follows the prose answer, never precedes it.
    workflow.add_edge("generate_answer", "visualize")
    workflow.add_edge("visualize", END)

    return workflow.compile()
