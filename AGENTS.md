# Study Coach Repo Rules

本文件只约束 `study-coach/` 仓库内的工作方式，不影响同级的 `JadeAI` 或 `HKBU_StudyCompanion`。

## Git 边界

- 这个 repo 只管理 `study-coach/`。
- 不把上层目录或兄弟项目纳入同一个仓库。
- 默认分支使用 `main`。

## 分支规则

- 分支命名使用：`feat/...`、`fix/...`、`docs/...`、`chore/...`、`exp/...`。
- 一个任务一条分支，不把无关改动混在一起。
- 没有明确要求时，不自动创建、切换、删除分支。

## Commit 规则

- 没有使用者明确要求时，不自动 commit。
- 一次 commit 只解决一个明确问题。
- 功能、配套测试、必要文档可以同一个 commit。
- 不顺手带入无关清理、格式化或重构。
- commit message 使用简洁英文前缀，例如：
  - `feat: add plan check-in endpoint`
  - `fix: correct quiz parser fallback`
  - `docs: sync roadmap with p3 state`

## 提交前验证

- 小改动：只验证变更点和直接相关部分。
- 后端行为改动：至少运行相关 `pytest`。
- 前端可见改动：至少运行 `pnpm build`。
- 跨前后端或核心流程改动：同时验证后端与前端。
- 没验证的部分必须明确说明原因。

## 文档同步

- 影响架构、数据流、接口、实验结论、产品行为时，必须同步更新 `docs/` 中对应文档。
- `README.md` 不要求每次都改，但不能长期落后于项目实际状态。
- 梳理项目时，优先读取：
  - `docs/ROADMAP.md`
  - `docs/ARCHITECTURE.md`
  - `docs/EVAL.md`
  - `design-system/MASTER.md`

## 入库与忽略

- 应提交：源码、测试、迁移、设计文档、精选 fixtures、必要截图。
- 不提交：`node_modules`、`dist`、`.venv`、cache、`.env`、本地 SQLite/Chroma 数据、临时日志、系统垃圾文件。
- `backend/app/eval/**/output` 这类实验运行输出默认不入库。
- 保留可复现实验所需的脚本、query 集、tests fixtures 和汇总结论文档。

## 数据库与实验

- 任何 schema 变化都必须带 Alembic migration。
- 涉及 migration、回填、删除、搬移时，必须说明风险、验证方式和回退方式。
- 阶段性实验优先保留：
  - 可复现代码
  - query / fixture
  - 汇总报告
- 大体积 raw outputs 只有在使用者明确要求存档时才入库。

## Portfolio 资产

- `docs/screenshots/`、`docs/superpowers/specs/`、`docs/superpowers/plans/` 可以入库。
- 只保留有展示或复盘价值的内容，不堆临时过程垃圾。

## Tag 建议

- 阶段完成后可按里程碑打 tag，例如：`p1`、`p2.0`、`p2.1`、`p3`。
- 没有使用者明确要求时，不自动打 tag。
