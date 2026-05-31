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
