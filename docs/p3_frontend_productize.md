# P3 update: shipping the product around the ablation

> A follow-up to `agent_loop_vs_deterministic.md` and `quiz_ablation_followup.md`. Two prior posts asked whether the agent loop pattern works; this one is about what to do with the answer when you ship the actual product.

## What I built

Seven views in Vue 3 over the existing FastAPI + LangGraph backend: Overview (hero dashboard), Chat, PlanTimeline, QuizAdaptive, MistakeBank, Library, Settings. The backend stayed byte-identical except for four new GET endpoints that surface data the frontend needs (`/api/plans/current`, `/api/documents`, `/api/mistakes/due`, `/api/mastery`). No schema migrations, no graph changes, no provider work. 213 backend tests pass, up from 202 — the +11 are all on the new endpoints and one repo method.

The visual direction is "Modern Dark Cinema": indigo primary on near-black neutrals, Inter for UI, JetBrains Mono for code, Noto Sans SC for CJK. All tokens live in a single Tailwind 4 `@theme` block in `design-system/MASTER.md`. I avoided shadcn-vue because the component count is well under threshold and adding a UI library before you need it tends to lock you into other people's decisions about which primitives matter.

The fidelity target was "bundle 2 mid": mermaid mindmap on the Plan view, chart.js radar on the Overview, SM-2 schedule numbers exposed on the Mistake Bank list rows, and an alignment-safety banner that gates the Quiz view when no corpus is loaded. The shipped Library also surfaces the persisted indexed-PDF list instead of only echoing the current upload, and the Mistake Bank now shows tracked mistakes even when their next review date is tomorrow rather than "due today". Everything draggable or animated was deferred. The point of P3 is the production seams, not motion design.

## Mode dispatch UX as the portfolio narrative

The dual-mode P2.2 / P2.3 finding lives in roughly 1300 lines of EVAL.md and two prior blog posts. None of that exists in a product UI. So the question for P3 was: how do you make a non-technical reader of the portfolio *feel* the agent-vs-deterministic trade-off without reading 12,000 words?

Three options were on the table. A global Settings-only toggle was too hidden — a reader who lands on `/quiz` will never go find it. A per-message toggle in the chat input was too noisy — it's an A/B switch on every send when the choice barely varies turn-to-turn. The shipped pattern is per-view default (from Settings, set independently for Plan and Quiz) plus a `<ModeChip>` in the view header that flips the mode for the *next* chat send and auto-reverts after `done`.

That last bit is the load-bearing detail: the chip is "sticky for one send, then snaps back." If you flip it and forget, your default behavior returns automatically. If you want to A/B both modes on the same question, you get a one-click compare without leaving the view. And because Settings persists the default, a reader who decides "deterministic is fine for Plan, agent_loop is essential for Quiz" can lock that in and never see the chip again.

The UX makes the latency cost legible too. On the Quiz view in agent_loop mode, the generate spinner runs 30 to 90 seconds (which is exactly what P2.3 measured). On deterministic, the same generate completes in 3 to 5. Same model. Same query. The product lets the user feel the trade-off the eval section is talking about.

## Operationalizing the alignment-safety finding

P2.3 §Finding 3 was the most interesting result of the whole ablation: well-aligned models running agent_loop with `retriever_search` returning `[]` will refuse to fabricate quiz questions, while deterministic prompt-only paths silently generate plausible-sounding questions from training-distribution priors. Refusal rates were 21-39% on the no-retriever pilot. That's a *good* property of the agent harness, surfaced by accident.

In production, you can't just let "Sorry, I can't make a quiz on a topic with no source material" appear as a raw chat bubble. The reader has no idea why they're being refused and clicks away. So `<EmptyCorpusBanner>` operationalizes the finding via two channels feeding one computed.

The pre-flight channel hits `GET /api/documents` on Quiz mount. If `chunks_count === 0` across all docs (or the user has zero docs), the banner renders before any chat request goes out. The user never sees the refusal text because the refusal never happens.

The in-flight channel matters when documents *exist* but the topic still misses (e.g. user asks for quiz on a topic the corpus doesn't cover). `parse.ts:looksLikeEmptyCorpusRefusal()` runs a regex over the accumulated streaming token buffer. If it matches, the quiz store sets `needsUpload = true` mid-stream and the banner appears. Both channels write to the same `needsUpload = docs.isEmpty || quiz.needsUpload` computed in `QuizAdaptive.vue`. One UI state, two detection paths, zero raw-refusal bubbles.

The regex is a known soft spot. Three months from now a different aligned model phrases its refusals differently and the in-flight detector silently degrades to pre-flight only. I tagged it `# cloud-adapt:` so when the BYOK cloud-model cut happens, the regex array is one of the first things to extend. The pre-flight channel doesn't have this brittleness — `chunks_count` is a number.

## Discipline calls I'd defend

Things I explicitly didn't do:

No Vitest. The chrome-devtools MCP verification flow — load a view, take a snapshot, take a screenshot, check the console — caught every real bug across 14 cuts in about 5 minutes per cut. Frontend unit testing in Vue 3 + Pinia + Tailwind is high-cost-low-yield at this scale; the bugs are visual or state-flow bugs that snapshot-driven verification catches better than mocked unit tests.

No shadcn-vue migration. 11 new components were enough for P3. Pulling in a UI library is a one-way door: once you've shaped your components around someone else's primitives, ripping it out costs more than starting clean. The rule from the roadmap: if component count grows beyond ~30, revisit.

No mobile UI. The spec said desktop-only for P3 with an explicit `<768px` banner. Mobile is a distinct product surface (touch targets, gesture nav, swipe-to-redo on Mistakes) and squeezing it into P3 would have meant cutting one of the four new views or shipping mobile half-broken. Better to ship 7 desktop views clean and earmark mobile as P4.

No drag-reorder on milestones. The Plan view is read-only with a check-in flow. P2.2 measured planner-agent latency at 5-30s; drag-reorder makes sense once you have a planner that can incrementally re-order without re-emitting the whole plan, and that's a P4 feature.

## What this proves

The full-stack portfolio artifact decomposes cleanly: backend ablation (P2.2 + P2.3) plus frontend product (P3). The backend work answered "does the agent loop pattern actually work on small Ollama models, or is it just a Sonnet/GPT-4 luxury?" The frontend work answered "if you take that ablation data seriously, what does the product look like?"

Each layer cites the other. The Settings → per-view ModeChip wiring is the productization of `agent_loop_vs_deterministic.md`. The EmptyCorpusBanner is the productization of `quiz_ablation_followup.md` §Finding 3. The Mistake Bank's SM-2 column is the productization of the P2.1-④ scheduler. None of it is decoration; every UI affordance maps to a measured backend behavior.

## What I'd do next

P4 priorities are documented in `ROADMAP.md`. Highest-value items: mobile UI; real streak + coverage axes on the radar chart (currently placeholders pending a sessions activity log); MCQ format hardening on the backend so the frontend parser doesn't have to be heuristic; Library auto-redirect after upload (the `?return=/quiz` query is already in the EmptyCorpusBanner click handler, the receiving end just needs to consume it); and BYOK cloud-model adaptation (5 marked sites in the spec, currently 1 implemented in `parse.ts`).

## Reproduce

```bash
cd study-coach/frontend
pnpm install
pnpm dev
# In another terminal:
cd study-coach/backend
uv sync
ollama pull gemma3:4b   # or any of the P2.2 tier
uv run uvicorn app.main:app --port 8000
# Visit http://localhost:5173/
```

---

*Sister posts: [Did I build a real agent?](./agent_loop_vs_deterministic.md) (P2.2) and [P2.3 update](./quiz_ablation_followup.md). Full P3 numbers at [EVAL.md §"P3 Frontend Productize"](./EVAL.md). Repo for this work: Study Coach (refactor of HKBU class project).*
