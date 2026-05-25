"""Cut ⑤c — Plan tool tests.

`update_study_plan` is a thin wrapper around PlanRepository.update_milestones —
the repo upsert is already covered by Cut ⑤b, so we only assert the contract here.
`generate_mindmap` is an LLM-driven tool; we stub the LLM and check the three
tolerant parsing tiers + fallback.
"""
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
