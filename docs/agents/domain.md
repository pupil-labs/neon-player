# Domain Docs

This repo uses a single-context domain documentation layout.

## Before exploring

- Read `GLOSSARY.md` at the repo root if it exists.
- Read relevant decisions in `docs/adr/` if that directory exists.

When these documents are absent, proceed silently. Domain modeling
creates them lazily when terms or decisions are resolved.

## File structure

- `GLOSSARY.md`: shared domain vocabulary.
- `docs/adr/NNNN-<decision-slug>.md`: architectural decisions.

## Use the glossary's vocabulary

Use defined domain terms in issue titles, proposals, hypotheses,
and test names.

If a needed concept is missing, reconsider whether it belongs to the
project's vocabulary or note the gap for domain modeling.

## Flag ADR conflicts

Explicitly identify proposals that contradict an existing ADR,
with the ADR reference and the reason to reconsider it.
