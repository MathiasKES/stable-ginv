# Project Agent Instructions

Read `docs/handover/README.md` first when starting a new chat or development
session. Follow its reading order before editing code.

Keep active handover files current-state only:

- Remove redundant, outdated, and superseded information when updating docs.
- Include everything a new agent needs to understand the active architecture,
  run current commands, avoid known pitfalls, verify changes, and continue
  pending work.
- Keep historical details in Git history unless they directly affect existing
  experiment results.

Do not modify reconstruction math, masking behavior, optimizer behavior, CLI
defaults, registry key inputs, or CSV columns during cleanup.

Do not edit modules imported by spawned workers while HPC experiments are
running. Newly spawned workers read current file contents from disk.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
