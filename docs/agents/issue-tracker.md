# Issue tracker: Local Markdown

Issues and specs live as Markdown files in `.scratch/`.

## Conventions

- One feature per directory: `.scratch/<feature-slug>/`.
- Spec: `.scratch/<feature-slug>/spec.md`.
- Implementation tickets: one file per ticket at
  `.scratch/<feature-slug>/issues/<NN>-<slug>.md`, numbered from `01`.
- Triage state: a `Status:` line near the top of each issue file.
  Use the roles in `triage-labels.md`.
- Comments and conversation history: append under `## Comments`.

## Publishing and fetching tickets

When a skill says "publish to the issue tracker", create the appropriate
file under `.scratch/<feature-slug>/`, creating directories as needed.

When a skill says "fetch the relevant ticket", read the referenced file.
Resolve ticket numbers within the relevant feature directory.

## Wayfinding operations

Used by `/wayfinder`.

- Map: `.scratch/<effort>/map.md`, containing Notes,
  Decisions-so-far, and Fog.
- Child ticket: `.scratch/<effort>/issues/NN-<slug>.md`, numbered
  from `01`, with the question in the body.
- Type: record `research`, `prototype`, `grilling`, or `task`
  in a `Type:` line.
- Blocking: record `Blocked by: NN, NN` near the top.
  A ticket is unblocked when every listed ticket is resolved.
- Frontier: scan for open, unblocked, unclaimed tickets;
  first by number wins.
- Claim: set `Status: claimed` and save before starting work.
- Resolve: append the answer under `## Answer`, set
  `Status: resolved`, and append a brief finding and ticket link
  to Decisions-so-far in the map.

Wayfinding uses `claimed` and `resolved` as execution statuses;
ordinary triage uses the roles in `triage-labels.md`.
