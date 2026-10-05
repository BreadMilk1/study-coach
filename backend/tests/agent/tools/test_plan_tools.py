"""Cut ⑤c — Plan tool tests.

`update_study_plan` is a thin wrapper around PlanRepository.update_milestones —
the repo upsert is already covered by Cut ⑤b, so we only assert the contract here.
`generate_mindmap` is an LLM-driven tool; we stub the LLM and check the three
tolerant parsing tiers + fallback.

Batch D1 adds content-block consumers: the stub LLM returns genuine AIMessage
objects whose `.content` is a block list, so the real tool (generate_mindmap ->
app.llm.content -> Mermaid / outline parser) is what is under test.
"""
import socket

import pytest
from langchain_core.messages import AIMessage
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.agent.tools.plan import generate_mindmap, update_study_plan
from app.agent.tools.schemas import Milestone
from app.db.models import Base
from app.db.repositories import GoalRepository, PlanRepository, UserRepository


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


class StubLLM:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.last_prompt: str | None = None

    async def ainvoke(self, messages, **_kwargs):
        self.last_prompt = messages[-1].content if messages else ""
        return AIMessage(content=self.response_text)


def test_update_study_plan_persists_milestones(session):
    user = UserRepository(session).get_or_create("fp-tool-plan")
    goal = GoalRepository(session).create(user_id=user.id, title="G")
    repo = PlanRepository(session)

    out = update_study_plan(
        goal_id=goal.id,
        milestones=[Milestone(title="Read §1", due_at="2026-05-30", done=False, topic="HyDE")],
        plan_repo=repo,
    )

    assert out.plan_id
    fetched = repo.get_by_goal(goal.id)
    assert fetched.milestones_json[0]["title"] == "Read §1"
    assert fetched.milestones_json[0]["due_at"] == "2026-05-30"


async def test_generate_mindmap_parses_fenced_mermaid():
    llm = StubLLM("""Here you go:
```mermaid
mindmap
  root((HyDE))
    Definition
    Steps
      Generate
      Embed
```
And outline:
- HyDE
  - Definition
  - Steps
""")
    out = await generate_mindmap(
        topic="HyDE",
        milestones=[Milestone(title="Read §1")],
        llm=llm,
    )

    assert "mindmap" in out.mermaid_src
    assert "root((HyDE))" in out.mermaid_src
    assert "HyDE" in out.markdown_outline


async def test_generate_mindmap_parses_bare_mermaid_without_fence():
    llm = StubLLM("""mindmap
  root((HyDE))
    Definition

Outline:
- HyDE
  - Definition
""")
    out = await generate_mindmap(
        topic="HyDE",
        milestones=[Milestone(title="Read §1")],
        llm=llm,
    )

    assert out.mermaid_src.startswith("mindmap")
    assert "HyDE" in out.markdown_outline


async def test_generate_mindmap_falls_back_to_outline_only_when_mermaid_unparseable():
    llm = StubLLM("Sorry, I can't draw a chart, but here's the outline:\n- HyDE\n  - Step 1")
    out = await generate_mindmap(
        topic="HyDE",
        milestones=[Milestone(title="Read §1")],
        llm=llm,
    )

    assert out.mermaid_src == ""
    # Outline is whatever survived parsing (we don't strictly require leading bullets,
    # only that some text reaches the user instead of crashing).
    assert out.markdown_outline.strip() != ""


async def test_generate_mindmap_llm_failure_returns_empty_mermaid():
    class CrashLLM:
        async def ainvoke(self, messages, **_kwargs):
            raise RuntimeError("ollama unreachable")

    out = await generate_mindmap(
        topic="HyDE",
        milestones=[Milestone(title="Read §1")],
        llm=CrashLLM(),
    )

    assert out.mermaid_src == ""
    assert "HyDE" in out.markdown_outline  # uses milestones as fallback outline


# ---------------------------------------------------------------------------
# Batch D1 — strict content-block text consumption in the mindmap tool.
# ---------------------------------------------------------------------------

_SENTINEL = "SENTINEL-DO-NOT-LEAK"
_MALFORMED_CONTENT_MESSAGE = "LLM response content is not a supported text shape"

_FENCED_MINDMAP = """Here you go:
```mermaid
mindmap
  root((HyDE))
    Definition
    Steps
      Generate
      Embed
```
And outline:
- HyDE
  - Definition
  - Steps
"""


@pytest.fixture
def no_network(monkeypatch):
    """Fail immediately on any real DNS / outbound socket attempt."""

    def deny(*_args, **_kwargs):
        raise AssertionError("D1 consumer test attempted real network access")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(socket, "getaddrinfo", deny)


class BlockLLM:
    """Returns one genuine `AIMessage` per call, each with block-list content."""

    def __init__(self, contents):
        self.contents = list(contents)
        self.calls = 0

    async def ainvoke(self, _messages, **_kwargs):
        if self.calls >= len(self.contents):
            raise AssertionError("BlockLLM exhausted")
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


async def test_generate_mindmap_consumes_fenced_mermaid_split_across_blocks(no_network):
    llm = BlockLLM([_split_across_blocks(_FENCED_MINDMAP, (10, 60))])

    out = await generate_mindmap(
        topic="HyDE",
        milestones=[Milestone(title="Read §1")],
        llm=llm,
    )

    assert out.mermaid_src.startswith("mindmap")
    assert "root((HyDE))" in out.mermaid_src
    assert "Generate" in out.mermaid_src
    assert "Definition" in out.markdown_outline
    assert _SENTINEL not in out.mermaid_src
    assert _SENTINEL not in out.markdown_outline


async def test_generate_mindmap_uses_visible_block_text_as_outline(no_network):
    llm = BlockLLM([[
        {"type": "thinking", "thinking": _SENTINEL},
        {"type": "text", "text": "- HyDE\n  - Step 1"},
    ]])

    out = await generate_mindmap(
        topic="HyDE",
        milestones=[Milestone(title="Read §1")],
        llm=llm,
    )

    assert out.mermaid_src == ""
    assert out.markdown_outline == "- HyDE\n  - Step 1"
    assert _SENTINEL not in out.markdown_outline


@pytest.mark.parametrize(
    "blocks",
    [
        [],
        [{"type": "unknown_vendor_block", "text": _SENTINEL}],
        [{"type": "thinking", "thinking": _SENTINEL}],
        [{"type": "text", "text": "  \n "}],
    ],
    ids=["no-blocks", "unknown-type", "skipped-only", "blank-text"],
)
async def test_generate_mindmap_falls_back_to_milestones_for_unusable_blocks(
    no_network, blocks
):
    llm = BlockLLM([blocks])

    out = await generate_mindmap(
        topic="HyDE",
        milestones=[Milestone(title="Read §1")],
        llm=llm,
    )

    assert out.mermaid_src == ""
    # Milestone-derived outline, never the block payload.
    assert "Read §1" in out.markdown_outline
    assert _SENTINEL not in out.markdown_outline
