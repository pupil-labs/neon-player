# Shared application types implementation

- [x] Read the Principles section of poteto-mode.
- [x] Phase A. Frame.
- [x] Phase B. Design the workflow.
- [x] Phase C. Run the loop.
- [x] Phase D. Keep the audit trail.
- [x] Phase E. Verify and hand back. Verification ran; full acceptance remains blocked as documented in implementation-report.md.

## Definition of done

No diagnostics from migrated shared contracts, no new suppressions, unchanged strict configuration, full pytest green, Ruff and formatting green. Independent remaining diagnostics are documented with before and after output.

## Workflow

1. Capture full mypy and pytest baselines.
2. Establish optional application access, required access, recording settings lifecycle, and typed plugin contracts.
3. Migrate numerical and plugin consumers in isolated copies.
4. Migrate Qt consumers and preserve fractional coordinates and export timestamps.
5. Integrate and run full checks. Review remaining errors by ownership.
6. Inspect diff and add focused behavior tests for lifecycle and framework boundaries.

## Throughput checkpoint

Baseline is 506 diagnostics across 31 of 43 checked files. Existing suite has 100 passing tests. Parallel units own numerical plugins and export/surface plugins. Coordinator owns shared providers, app, settings, and Qt. Architecture exploration is skipped because the spec pins the existing shape and the change is a reversible contract migration.
