"""Versioned contracts for the Study Coach Learning Run harness."""

from .contracts import (
    CalibrationCandidate,
    CalibrationTaskInput,
    CandidateArtifact,
    CorpusChunk,
    CorpusSnapshot,
    ExperimentDefinition,
    PromptDefinition,
    ResolvedRunDefinition,
    RunManifest,
    ScorerBundle,
    ScorerComponent,
    TaskCase,
    canonical_hash,
    canonical_json_bytes,
)

__all__ = [
    "CorpusChunk",
    "CorpusSnapshot",
    "CalibrationCandidate",
    "CalibrationTaskInput",
    "CandidateArtifact",
    "ExperimentDefinition",
    "PromptDefinition",
    "RegistryError",
    "ResolvedRunDefinition",
    "RunManifest",
    "ScorerBundle",
    "ScorerComponent",
    "TaskCase",
    "TaskRegistry",
    "canonical_hash",
    "canonical_json_bytes",
]


def __getattr__(name: str):
    if name in {"RegistryError", "TaskRegistry"}:
        from .registry import RegistryError, TaskRegistry

        return {"RegistryError": RegistryError, "TaskRegistry": TaskRegistry}[name]
    raise AttributeError(name)
