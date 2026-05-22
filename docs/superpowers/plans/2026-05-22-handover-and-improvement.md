# Handover Files + Code Improvement Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce three audience-specific handover files and a two-phase code improvement plan for the stable-ginv gradient inversion research codebase.

**Architecture:** Each handover file is self-contained markdown targeting a specific reader (researcher, reviewer, thesis collaborator). The improvement plan is a structured two-phase document separate from the handover files.

**Tech Stack:** Python 3.13, PyTorch, torchvision, scikit-image, scipy, NumPy, Matplotlib

---

## File Map

| File | Purpose |
|------|---------|
| `docs/handover/HANDOVER_RESEARCHER.md` | For a developer/researcher continuing the work |
| `docs/handover/HANDOVER_REVIEWER.md` | For a code-review agent doing targeted cleanup |
| `docs/handover/HANDOVER_COLLABORATOR.md` | For a fresh thesis collaborator |
| `docs/IMPROVEMENT_PLAN.md` | Two-phase code improvement plan |

---

### Task 1: Write HANDOVER_RESEARCHER.md

**Files:**
- Create: `docs/handover/HANDOVER_RESEARCHER.md`

- [ ] **Step 1:** Write file with sections: Project Overview, Module Map, Entry Points, Configuration Reference, Masking Modes, Known Quirks, Recent Changes, Open Questions, Environment Setup, Output Interpretation
- [ ] **Step 2:** Commit

```bash
git add docs/handover/HANDOVER_RESEARCHER.md
git commit -m "docs: add researcher handover file"
```

---

### Task 2: Write HANDOVER_REVIEWER.md

**Files:**
- Create: `docs/handover/HANDOVER_REVIEWER.md`

- [ ] **Step 1:** Write file with sections: Codebase Scope, Priority Files for Review, Specific Issues to Address, Do Not Change, Patterns to Follow, How to Verify
- [ ] **Step 2:** Commit

```bash
git add docs/handover/HANDOVER_REVIEWER.md
git commit -m "docs: add code reviewer handover file"
```

---

### Task 3: Write HANDOVER_COLLABORATOR.md

**Files:**
- Create: `docs/handover/HANDOVER_COLLABORATOR.md`

- [ ] **Step 1:** Write file with sections: Research Background, The Attack Model, How Masking Defends, Environment Setup, Running Your First Experiment, Interpreting Results, Paper-to-Code Map, Dataset Setup
- [ ] **Step 2:** Commit

```bash
git add docs/handover/HANDOVER_COLLABORATOR.md
git commit -m "docs: add thesis collaborator handover file"
```

---

### Task 4: Write IMPROVEMENT_PLAN.md

**Files:**
- Create: `docs/IMPROVEMENT_PLAN.md`

- [ ] **Step 1:** Write Phase 1 (quick wins): split Misc_functions.py, add docstrings, unify naming, deduplicate constants, clean dead code
- [ ] **Step 2:** Write Phase 2 (structural): package layout, config dataclasses, shared CLI, unit tests, type hints
- [ ] **Step 3:** Commit

```bash
git add docs/IMPROVEMENT_PLAN.md
git commit -m "docs: add two-phase code improvement plan"
```
