"""Cut ⑤e — planner_node tests (GENERATE + CHECK-IN).

Mirrors test_quiz_master.py: factory built with real in-memory SQLite repos,
LLM stubbed. Exercises both decide() paths + edge cases.

Batch D1 adds content-block consumers: the stub LLM returns genuine AIMessage
objects whose `.content` is a block list, so the real node (planner ->
app.llm.content -> milestone parser -> repository) is what is under test. Only
the model is faked.
"""
import json
import socket
from datetime import datetime
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.agent.planner import build_planner
from app.db.models import Base
from app.db.repositories import (
    GoalRepository,
    MasteryRepository,
    MistakeRepository,
    PlanRepository,
    UserRepository,
)

_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "app" / "agent" / "prompts"


_GEN_JSON = """[
  {"title": "Read HyDE §1-§3", "due_at": "2026-05-25", "done": false, "topic": "HyDE"},
  {"title": "Implement HyDEGenerator", "due_at": "2026-05-28", "done": false, "topic": "HyDE"},
  {"title": "Compare HyDE vs BM25", "due_at": "2026-06-01", "done": false, "topic": "HyDE"}
]"""

_CHECK_IN_JSON = """[
  {"title": "Read HyDE §1-§3", "due_at": "2026-05-25", "done": true, "topic": "HyDE"},
  {"title": "Implement HyDEGenerator", "due_at": "2026-05-24", "done": false, "topic": "HyDE"},
  {"title": "Compare HyDE vs BM25", "due_at": "2026-06-01", "done": false, "topic": "HyDE"}
]"""


class StubPlannerLLM:
    def __init__(self, response_text: str = _GEN_JSON):
        self.response_text = response_text
        self.last_prompt: str | None = None

    async def ainvoke(self, messages, **_kwargs):
        self.last_prompt = messages[-1].content if messages else ""
        return AIMessage(content=self.response_text)


class StubRetriever:
    def __init__(self, chunks=None):
        self.chunks = chunks or []
        self.last_query: str | None = None

    def search(self, query, top_k=5):
        self.last_query = query
        return self.chunks[:top_k]


@pytest.fixture
def session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def _build_node(session, llm=None, retriever=None):
    return build_planner(
        llm=llm or StubPlannerLLM(),
        plan_repo=PlanRepository(session),
        goal_repo=GoalRepository(session),
        mastery_repo=MasteryRepository(session),
        mistake_repo=MistakeRepository(session),
        retriever=retriever,
        now_fn=lambda: datetime(2026, 5, 22, 12, 0),
    )


def test_planner_check_in_prompt_requires_id_and_topic_id_preservation():
    prompt = (_PROMPTS_DIR / "planner_check_in.txt").read_text(encoding="utf-8")

    assert "id/topic_id" in prompt


async def test_planner_generate_creates_plan_and_sets_active_plan_id(session):
    user = UserRepository(session).get_or_create("fp-plan-gen")
    node = _build_node(session)

    update = await node({
        "messages": [HumanMessage(content="帮我做学习计划 on HyDE")],
        "user_id": user.id,
    })

    assert update["plan_action"] == "generate"
    assert update["active_plan_id"]
    # Plan row created
    plan_repo = PlanRepository(session)
    goal = GoalRepository(session).list_active_for_user(user.id)[0]
    saved = plan_repo.get_by_goal(goal.id)
    assert saved is not None
    assert len(saved.milestones_json) == 3
    # Output text mentions milestones
    text = update["messages"][0].content
    assert "Read HyDE" in text


async def test_planner_generate_skips_mindmap_without_keyword(session):
    user = UserRepository(session).get_or_create("fp-plan-no-mm")
    node = _build_node(session)

    update = await node({
        "messages": [HumanMessage(content="帮我做学习计划 on HyDE")],
        "user_id": user.id,
    })

    text = update["messages"][0].content
    assert "mindmap" not in text.lower()
    assert "```" not in text  # no mermaid fence


async def test_planner_generate_calls_mindmap_on_keyword(session):
    user = UserRepository(session).get_or_create("fp-plan-mm")

    # Sequence two stub responses: first call = milestones JSON, second = mindmap text.
    class TwoStepLLM:
        def __init__(self):
            self.calls = 0

        async def ainvoke(self, messages, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content=_GEN_JSON)
            return AIMessage(content="```mermaid\nmindmap\n  root((HyDE))\n```\n- HyDE")

    llm = TwoStepLLM()
    node = _build_node(session, llm=llm)

    update = await node({
        "messages": [HumanMessage(content="帮我做学习计划 on HyDE 画脑图")],
        "user_id": user.id,
    })

    assert llm.calls == 2
    text = update["messages"][0].content
    assert "mindmap" in text.lower()


async def test_planner_generate_uses_retriever_chunks_in_prompt(session):
    user = UserRepository(session).get_or_create("fp-plan-rag")
    retriever = StubRetriever(chunks=[
        {"chunk_id": "t7:p1", "content": "HyDE = Hypothetical Document Embedding."},
    ])
    llm = StubPlannerLLM()
    node = _build_node(session, llm=llm, retriever=retriever)

    await node({
        "messages": [HumanMessage(content="帮我做学习计划 on HyDE")],
        "user_id": user.id,
    })

    assert retriever.last_query == "HyDE"
    assert "Hypothetical Document Embedding" in llm.last_prompt


async def test_planner_check_in_adjusts_existing_plan(session):
    user = UserRepository(session).get_or_create("fp-plan-ci")
    goal = GoalRepository(session).create(user_id=user.id, title="G")
    plan_repo = PlanRepository(session)
    initial = plan_repo.create(
        goal_id=goal.id,
        milestones_json=[
            {"title": "Read HyDE §1-§3", "due_at": "2026-05-25", "done": False, "topic": "HyDE"},
            {"title": "Implement HyDEGenerator", "due_at": "2026-05-20", "done": False, "topic": "HyDE"},
            {"title": "Compare HyDE vs BM25", "due_at": "2026-06-01", "done": False, "topic": "HyDE"},
        ],
    )
    llm = StubPlannerLLM(_CHECK_IN_JSON)
    node = _build_node(session, llm=llm)

    update = await node({
        "messages": [HumanMessage(content="Read HyDE §1-§3 完成了，Implement HyDEGenerator 需要延期")],
        "user_id": user.id,
        "active_plan_id": initial.id,
        "mastery_scores": {"BM25": 0.2, "HyDE": 0.6},
    })

    assert update["plan_action"] == "check_in"
    refreshed = plan_repo.get_by_goal(goal.id)
    assert len(refreshed.milestones_json) == 3
    assert refreshed.milestones_json[0]["done"] is True
    assert refreshed.milestones_json[1]["due_at"] == "2026-05-24"
    assert [m["title"] for m in refreshed.milestones_json] == [
        "Read HyDE §1-§3",
        "Implement HyDEGenerator",
        "Compare HyDE vs BM25",
    ]
    text = update["messages"][0].content
    assert "Done:" in text or "进度" in text  # progress card surfaced


async def test_planner_check_in_with_unparseable_llm_output_keeps_plan_and_notes_skip(session):
    user = UserRepository(session).get_or_create("fp-plan-ci-bad")
    goal = GoalRepository(session).create(user_id=user.id, title="G")
    plan_repo = PlanRepository(session)
    initial = plan_repo.create(
        goal_id=goal.id,
        milestones_json=[{"title": "Read HyDE §1-§3", "due_at": "2026-05-25", "done": False, "topic": "HyDE"}],
    )
    llm = StubPlannerLLM(response_text="Sorry, I cannot do that today.")
    node = _build_node(session, llm=llm)

    update = await node({
        "messages": [HumanMessage(content="进度怎么样了")],
        "user_id": user.id,
        "active_plan_id": initial.id,
        "mastery_scores": {},
    })

    refreshed = plan_repo.get_by_goal(goal.id)
    assert len(refreshed.milestones_json) == 1  # unchanged
    text = update["messages"][0].content
    assert "Auto-adjust skipped" in text


async def test_planner_check_in_falls_back_to_generate_when_plan_missing(session):
    """active_plan_id set but plan was deleted externally → recover via GENERATE path."""
    user = UserRepository(session).get_or_create("fp-plan-recover")
    node = _build_node(session)

    update = await node({
        "messages": [HumanMessage(content="帮我做学习计划 on HyDE")],
        "user_id": user.id,
        "active_plan_id": "deadbeef-not-in-db",
    })

    # GENERATE took over; new plan created
    assert update["plan_action"] == "generate"
    assert update["active_plan_id"] != "deadbeef-not-in-db"


async def test_planner_force_generate_when_create_keyword_present_with_active_plan(session):
    """Fix A: 帮我做学习计划 on X with active_plan_id → GENERATE (overwrites), not CHECK-IN."""
    user = UserRepository(session).get_or_create("fp-plan-force-gen")
    goal = GoalRepository(session).create(user_id=user.id, title="G")
    plan_repo = PlanRepository(session)
    plan_repo.create(
        goal_id=goal.id,
        milestones_json=[{"title": "old M1", "done": False, "topic": "OldTopic"}],
    )
    node = _build_node(session)  # StubPlannerLLM returns _GEN_JSON (3 fresh HyDE milestones)

    update = await node({
        "messages": [HumanMessage(content="帮我做学习计划 on HyDE")],
        "user_id": user.id,
        "active_plan_id": "stale-id-but-old-plan-exists",
    })

    assert update["plan_action"] == "generate"
    refreshed = plan_repo.get_by_goal(goal.id)
    # old plan overwritten (upsert), now has the 3 fresh milestones
    assert len(refreshed.milestones_json) == 3
    assert refreshed.milestones_json[0]["title"] == "Read HyDE §1-§3"


async def test_planner_check_in_progress_count_matches_final_milestone_count(session):
    """Fix C: Done: X / Y where Y == len(Updated Plan list)."""
    user = UserRepository(session).get_or_create("fp-plan-ci-count")
    goal = GoalRepository(session).create(user_id=user.id, title="G")
    plan_repo = PlanRepository(session)
    initial = plan_repo.create(
        goal_id=goal.id,
        milestones_json=[
            {"title": "Old A", "done": False, "topic": "HyDE"},
            {"title": "Old B", "done": False, "topic": "HyDE"},
        ],
    )
    check_in_json = """[
      {"title": "Old A", "done": true, "topic": "HyDE"},
      {"title": "Old B", "done": false, "topic": "HyDE"}
    ]"""
    llm = StubPlannerLLM(check_in_json)
    node = _build_node(session, llm=llm)

    update = await node({
        "messages": [HumanMessage(content="进度怎么样了")],
        "user_id": user.id,
        "active_plan_id": initial.id,
    })

    text = update["messages"][0].content
    # Find the "Done: X / Y" line
    done_line = next((line for line in text.splitlines() if line.startswith("- Done:")), None)
    assert done_line is not None
    # Y must equal the number of milestones in the Updated Plan section
    refreshed = plan_repo.get_by_goal(goal.id)
    expected_total = len(refreshed.milestones_json)
    assert f"/ {expected_total}" in done_line, f"expected '/ {expected_total}' in {done_line!r}"


# ---------------------------------------------------------------------------
# Batch D1 — strict content-block text consumption on the deterministic planner.
#
# The deterministic planner is the production GENERATE / CHECK-IN path. Each
# case drives it with a genuine `AIMessage` whose `.content` is a block list;
# only the model is faked and the real helper / parser / repository run.
# ---------------------------------------------------------------------------

_SENTINEL = "SENTINEL-DO-NOT-LEAK"
_MALFORMED_CONTENT_MESSAGE = "LLM response content is not a supported text shape"
_EMPTY_CONTENT_MESSAGE = "LLM response text has no non-whitespace body"
_COULD_NOT_DRAFT = "Couldn't draft a plan on 'HyDE'. Try a clearer goal."


@pytest.fixture
def no_network(monkeypatch):
    """Fail immediately on any real DNS / outbound socket attempt."""

    def deny(*_args, **_kwargs):
        raise AssertionError("D1 consumer test attempted real network access")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(socket, "getaddrinfo", deny)


class BlockPlannerLLM:
    """Returns one genuine `AIMessage` per call, each with block-list content."""

    def __init__(self, contents):
        self.contents = list(contents)
        self.calls = 0

    async def ainvoke(self, _messages, **_kwargs):
        if self.calls >= len(self.contents):
            raise AssertionError("BlockPlannerLLM exhausted")
        content = self.contents[self.calls]
        self.calls += 1
        return AIMessage(content=content)


def _split_across_blocks(payload: str, cuts: tuple[int, ...]) -> list:
    """Thinking marker + text blocks (and one bare str) carrying `payload`."""
    parts: list = [{"type": "thinking", "thinking": _SENTINEL}]
    previous = 0
    for index, cut in enumerate(cuts):
        segment = payload[previous:cut]
        parts.append(segment if index == 1 else {"type": "text", "text": segment})
        previous = cut
    parts.append({"type": "text", "text": payload[previous:]})
    return parts


def _seed_plan(session, fingerprint: str):
    user = UserRepository(session).get_or_create(fingerprint)
    goal = GoalRepository(session).create(user_id=user.id, title="G")
    plan = PlanRepository(session).create(
        goal_id=goal.id,
        milestones_json=[
            {"title": "Read HyDE §1-§3", "due_at": "2026-05-25", "done": False, "topic": "HyDE"},
            {"title": "Implement HyDEGenerator", "due_at": "2026-05-20", "done": False, "topic": "HyDE"},
            {"title": "Compare HyDE vs BM25", "due_at": "2026-06-01", "done": False, "topic": "HyDE"},
        ],
    )
    return user, goal, plan


# --- GENERATE ---------------------------------------------------------------


async def test_planner_generate_consumes_milestone_json_split_across_text_blocks(
    session, no_network
):
    user = UserRepository(session).get_or_create("fp-d1-gen-blocks")
    llm = BlockPlannerLLM([_split_across_blocks(_GEN_JSON, (40, 140))])
    node = _build_node(session, llm=llm)

    update = await node({
        "messages": [HumanMessage(content="帮我做学习计划 on HyDE")],
        "user_id": user.id,
    })

    assert update["plan_action"] == "generate"
    assert update["active_plan_id"]
    goal = GoalRepository(session).list_active_for_user(user.id)[0]
    saved = PlanRepository(session).get_by_goal(goal.id)
    assert saved is not None
    assert [m["title"] for m in saved.milestones_json] == [
        "Read HyDE §1-§3",
        "Implement HyDEGenerator",
        "Compare HyDE vs BM25",
    ]
    text = update["messages"][0].content
    assert isinstance(text, str)
    assert "Read HyDE" in text
    assert _SENTINEL not in text


@pytest.mark.parametrize(
    "blocks",
    [
        [],  # control: already handled before D1 (no blocks at all)
        [{"type": "thinking", "thinking": _SENTINEL}],
        [{"type": "text", "text": "   "}, {"type": "text", "text": "\n\n"}],
        [{"type": "text-plain", "text": _SENTINEL}],  # published skipped type
    ],
)
async def test_planner_generate_without_usable_body_keeps_friendly_failure(
    session, no_network, blocks
):
    user = UserRepository(session).get_or_create("fp-d1-gen-empty")
    llm = BlockPlannerLLM([blocks])
    node = _build_node(session, llm=llm)

    update = await node({
        "messages": [HumanMessage(content="帮我做学习计划 on HyDE")],
        "user_id": user.id,
    })

    assert update["messages"][0].content == _COULD_NOT_DRAFT
    assert "active_plan_id" not in update
    assert _SENTINEL not in update["messages"][0].content
    # The goal may already have been created before the model call, but no plan.
    active = GoalRepository(session).list_active_for_user(user.id)
    assert active
    assert PlanRepository(session).get_by_goal(active[0].id) is None


async def test_planner_generate_with_non_milestone_text_blocks_keeps_parse_failure(
    session, no_network
):
    user = UserRepository(session).get_or_create("fp-d1-gen-nonjson")
    llm = BlockPlannerLLM([
        [{"type": "thinking", "thinking": _SENTINEL},
         {"type": "text", "text": "Sorry, I can only help with study plans."}],
    ])
    node = _build_node(session, llm=llm)

    update = await node({
        "messages": [HumanMessage(content="帮我做学习计划 on HyDE")],
        "user_id": user.id,
    })

    assert update["messages"][0].content == _COULD_NOT_DRAFT
    assert "active_plan_id" not in update
    active = GoalRepository(session).list_active_for_user(user.id)
    assert active
    assert PlanRepository(session).get_by_goal(active[0].id) is None


@pytest.mark.parametrize(
    "bad_block",
    [
        {"type": "unknown_vendor_block", "text": _SENTINEL},
        {"text": _SENTINEL},  # missing type
        {"type": "text", "text": 42},  # text is not a str
    ],
    ids=["unknown-type", "missing-type", "non-str-text"],
)
async def test_planner_generate_rejects_malformed_blocks_without_drafting_a_plan(
    session, no_network, bad_block
):
    user = UserRepository(session).get_or_create("fp-d1-gen-bad")
    llm = BlockPlannerLLM([[{"type": "thinking", "thinking": _SENTINEL}, bad_block]])
    node = _build_node(session, llm=llm)

    with pytest.raises(ValueError) as excinfo:
        await node({
            "messages": [HumanMessage(content="帮我做学习计划 on HyDE")],
            "user_id": user.id,
        })

    assert str(excinfo.value) == _MALFORMED_CONTENT_MESSAGE
    assert _SENTINEL not in str(excinfo.value)
    # The malformed payload must not be turned into a plan that looks drafted.
    active = GoalRepository(session).list_active_for_user(user.id)
    assert active  # goal creation happens before the model call
    assert PlanRepository(session).get_by_goal(active[0].id) is None


# --- CHECK-IN ---------------------------------------------------------------


async def test_planner_check_in_consumes_check_in_json_split_across_text_blocks(
    session, no_network
):
    user, goal, plan = _seed_plan(session, "fp-d1-ci-blocks")
    llm = BlockPlannerLLM([_split_across_blocks(_CHECK_IN_JSON, (30, 150))])
    node = _build_node(session, llm=llm)

    update = await node({
        "messages": [HumanMessage(content="进度怎么样了")],
        "user_id": user.id,
        "active_plan_id": plan.id,
    })

    assert update["plan_action"] == "check_in"
    refreshed = PlanRepository(session).get_by_goal(goal.id)
    assert refreshed.id == plan.id
    assert [m["title"] for m in refreshed.milestones_json] == [
        "Read HyDE §1-§3",
        "Implement HyDEGenerator",
        "Compare HyDE vs BM25",
    ]
    assert refreshed.milestones_json[0]["done"] is True
    assert refreshed.milestones_json[1]["due_at"] == "2026-05-24"
    text = update["messages"][0].content
    assert "Auto-adjust skipped" not in text
    assert _SENTINEL not in text


@pytest.mark.parametrize(
    "blocks",
    [
        [{"type": "unknown_vendor_block", "text": _SENTINEL}],
        [{"type": "thinking", "thinking": _SENTINEL}],
        [{"type": "text", "text": "Sorry, I can't adjust that right now."}],
    ],
    ids=["malformed-blocks", "skipped-only-blocks", "non-milestone-text"],
)
async def test_planner_check_in_falls_back_without_updating_the_plan(
    session, no_network, blocks
):
    user, goal, plan = _seed_plan(session, "fp-d1-ci-fallback")
    before = PlanRepository(session).get_by_goal(goal.id)
    before_snapshot = json.dumps(
        {
            "id": before.id,
            "milestones": before.milestones_json,
            "updated_at": before.updated_at.isoformat(),
        },
        sort_keys=True,
    )
    llm = BlockPlannerLLM([blocks])
    node = _build_node(session, llm=llm)

    update = await node({
        "messages": [HumanMessage(content="进度怎么样了")],
        "user_id": user.id,
        "active_plan_id": plan.id,
    })

    assert update["plan_action"] == "check_in"
    assert update["active_plan_id"] == plan.id
    after = PlanRepository(session).get_by_goal(goal.id)
    after_snapshot = json.dumps(
        {
            "id": after.id,
            "milestones": after.milestones_json,
            "updated_at": after.updated_at.isoformat(),
        },
        sort_keys=True,
    )
    # Same plan row, same milestones, same updated_at: no update, no new plan.
    assert after_snapshot == before_snapshot
    text = update["messages"][0].content
    assert "Auto-adjust skipped" in text
    assert "**Progress Check-in**" in text
    assert _SENTINEL not in text


# --- GENERATE + mindmap keyword --------------------------------------------


async def test_planner_generate_keeps_persisted_plan_when_mindmap_blocks_are_malformed(
    session, no_network
):
    user = UserRepository(session).get_or_create("fp-d1-mm-bad")
    llm = BlockPlannerLLM([
        _GEN_JSON,
        [{"type": "unknown_vendor_block", "text": _SENTINEL}],
    ])
    node = _build_node(session, llm=llm)

    update = await node({
        "messages": [HumanMessage(content="帮我做学习计划 on HyDE 画脑图")],
        "user_id": user.id,
    })

    # The plan the planner already persisted must survive a failing mindmap tool.
    goal = GoalRepository(session).list_active_for_user(user.id)[0]
    saved = PlanRepository(session).get_by_goal(goal.id)
    assert saved is not None
    assert len(saved.milestones_json) == 3
    text = update["messages"][0].content
    assert "```mermaid" not in text
    assert _SENTINEL not in text
    # The tool's own soft fallback (milestone-derived outline, empty mermaid).
    assert "**Outline**" in text
    assert "Read HyDE §1-§3" in text
