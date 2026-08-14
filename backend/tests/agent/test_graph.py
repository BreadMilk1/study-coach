from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from app.agent.graph import build_graph
from app.agent.tutor_attempt import TutorAttemptEngine


class StubLLM:
    def __init__(self, response_text: str = "Stub answer based on the context."):
        self.response_text = response_text
        self.last_prompt: str | None = None

    def invoke(self, messages, **_kwargs):
        self.last_prompt = messages[-1].content if messages else ""
        return AIMessage(content=self.response_text)

    async def astream(self, messages, **_kwargs):
        self.last_prompt = messages[-1].content if messages else ""
        yield AIMessageChunk(content=self.response_text)


class StubRetriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.last_query: str | None = None

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        self.last_query = query
        return self.chunks[:top_k]


class RecordingTutorAttemptEngine:
    def __init__(self):
        self.calls: list[dict] = []
        self.delegate = TutorAttemptEngine()

    async def answer(self, **kwargs):
        self.calls.append({
            "question": kwargs["question"],
            "attempt_config": kwargs["attempt_config"],
        })
        return await self.delegate.answer(**kwargs)


async def test_graph_retrieves_then_answers_using_retriever_chunks():
    chunks = [
        {"chunk_id": "a:1:0", "content": "HyDE generates a hypothetical answer first.",
         "source": "a.pdf", "page": 1, "score": 0.9},
        {"chunk_id": "b:2:0", "content": "BM25 is a lexical retrieval method.",
         "source": "b.pdf", "page": 2, "score": 0.7},
    ]
    retriever = StubRetriever(chunks)
    llm = StubLLM("HyDE is a query rewriting technique.")
    graph = build_graph(retriever=retriever, llm=llm)

    result = await graph.ainvoke({"messages": [HumanMessage(content="What is HyDE?")]})

    assert retriever.last_query == "What is HyDE?"
    assert "HyDE" in llm.last_prompt
    assert "BM25" in llm.last_prompt
    assert any(isinstance(m, AIMessage) for m in result["messages"])

    # Citation contract per ARCHITECTURE.md TypedDict: 5 fields incl. span.
    assert len(result["citations"]) == 2
    for i, c in enumerate(result["citations"]):
        assert c["chunk_id"] == chunks[i]["chunk_id"]
        assert c["source"] == chunks[i]["source"]
        assert c["page"] == chunks[i]["page"]
        assert c["span_start"] == 0
        assert c["span_end"] == len(chunks[i]["content"])


async def test_graph_returns_empty_citations_when_retriever_finds_nothing():
    retriever = StubRetriever([])
    llm = StubLLM("I don't have enough information.")
    graph = build_graph(retriever=retriever, llm=llm)

    result = await graph.ainvoke({"messages": [HumanMessage(content="Anything?")]})

    assert result["citations"] == []
    assert any(isinstance(m, AIMessage) for m in result["messages"])


async def test_graph_prompt_injects_source_filename_and_page_per_chunk():
    chunks = [
        {"chunk_id": "a:5:0", "content": "HyDE generates a hypothetical answer.",
         "source": "Topic 7 - Taming the Model.pdf", "page": 5, "score": 0.9},
    ]
    retriever = StubRetriever(chunks)
    llm = StubLLM()
    graph = build_graph(retriever=retriever, llm=llm)

    await graph.ainvoke({"messages": [HumanMessage(content="What is HyDE?")]})

    assert llm.last_prompt is not None
    assert "Topic 7 - Taming the Model.pdf" in llm.last_prompt
    assert ("p.5" in llm.last_prompt) or ("page 5" in llm.last_prompt)


async def test_graph_prompt_includes_grounded_answer_instruction():
    retriever = StubRetriever([])
    llm = StubLLM()
    graph = build_graph(retriever=retriever, llm=llm)

    await graph.ainvoke({"messages": [HumanMessage(content="anything")]})

    assert llm.last_prompt is not None
    grounded_phrases = [
        "don't know",
        "do not know",
        "say so",
        "no relevant",
        "not in the source",
        "cannot answer",
        "insufficient",
    ]
    lowered = llm.last_prompt.lower()
    assert any(p in lowered for p in grounded_phrases), (
        f"prompt missing grounded-answer instruction: {llm.last_prompt[:200]}"
    )


# --- P2.1-① multi-node graph tests ---------------------------------------


async def test_router_node_writes_intent_into_state_for_quiz_query():
    retriever = StubRetriever([])
    llm = StubLLM()
    graph = build_graph(retriever=retriever, llm=llm)

    result = await graph.ainvoke({"messages": [HumanMessage(content="测我一下 HyDE")]})

    assert result.get("intent") == "quiz"


async def test_router_node_writes_intent_into_state_for_plan_query():
    retriever = StubRetriever([])
    llm = StubLLM()
    graph = build_graph(retriever=retriever, llm=llm)

    result = await graph.ainvoke({"messages": [HumanMessage(content="帮我做学习计划")]})

    assert result.get("intent") == "plan"


async def test_graph_routes_quiz_query_to_quiz_stub():
    retriever = StubRetriever([
        {"chunk_id": "x:1:0", "content": "should not be returned",
         "source": "x.pdf", "page": 1, "score": 0.9},
    ])
    llm = StubLLM("real tutor answer")
    graph = build_graph(retriever=retriever, llm=llm)

    result = await graph.ainvoke({"messages": [HumanMessage(content="quiz me on RAG")]})

    # Quiz branch must NOT touch retriever (saves cost)
    assert retriever.last_query is None
    # Last AIMessage carries stub marker, not the real tutor answer
    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert ai_msgs, "expected at least one AIMessage in result"
    assert "P2.1-④" in ai_msgs[-1].content
    assert "real tutor answer" not in ai_msgs[-1].content
    # Stub still emits an empty citations list to keep frontend contract uniform
    assert result.get("citations") == []


async def test_graph_routes_plan_query_to_plan_stub():
    retriever = StubRetriever([
        {"chunk_id": "x:1:0", "content": "should not be returned",
         "source": "x.pdf", "page": 1, "score": 0.9},
    ])
    llm = StubLLM("real tutor answer")
    graph = build_graph(retriever=retriever, llm=llm)

    result = await graph.ainvoke({"messages": [HumanMessage(content="设定我的学习计划")]})

    assert retriever.last_query is None
    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert ai_msgs
    assert "P2.1-⑤" in ai_msgs[-1].content
    assert "real tutor answer" not in ai_msgs[-1].content
    assert result.get("citations") == []


async def test_graph_routes_tutor_query_through_retrieve_and_answer():
    chunks = [
        {"chunk_id": "a:1:0", "content": "HyDE rewrites queries.",
         "source": "a.pdf", "page": 1, "score": 0.9},
    ]
    retriever = StubRetriever(chunks)
    llm = StubLLM("HyDE is a technique.")
    graph = build_graph(retriever=retriever, llm=llm)

    result = await graph.ainvoke({"messages": [HumanMessage(content="What is HyDE?")]})

    assert retriever.last_query == "What is HyDE?"
    assert result.get("intent") == "tutor"
    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert ai_msgs and ai_msgs[-1].content == "HyDE is a technique."
    assert result.get("citations") and len(result["citations"]) == 1


async def test_graph_tutor_adapter_maps_one_attempt_candidate_without_changing_state_contract():
    chunks = [
        {"chunk_id": "a:1:0", "content": "HyDE rewrites queries.",
         "source": "a.pdf", "page": 1, "score": 0.9},
    ]
    retriever = StubRetriever(chunks)
    llm = StubLLM("HyDE is a technique.")
    attempt_engine = RecordingTutorAttemptEngine()
    graph = build_graph(
        retriever=retriever,
        llm=llm,
        tutor_attempt_engine=attempt_engine,
    )

    result = await graph.ainvoke({"messages": [HumanMessage(content="What is HyDE?")]})

    assert len(attempt_engine.calls) == 1
    assert attempt_engine.calls[0]["question"] == "What is HyDE?"
    assert attempt_engine.calls[0]["attempt_config"].top_k == 5
    assert attempt_engine.calls[0]["attempt_config"].retrieval_seconds is None
    assert attempt_engine.calls[0]["attempt_config"].generation_seconds is None
    assert result["messages"][-1].content == "HyDE is a technique."
    assert result["citations"] == [
        {
            "chunk_id": "a:1:0",
            "source": "a.pdf",
            "page": 1,
            "span_start": 0,
            "span_end": len(chunks[0]["content"]),
        }
    ]
    assert result["last_context"] == (
        "[1] a.pdf p.1: HyDE rewrites queries."
    )
