# Make shared application types accurate under strict mypy

Status: ready-for-agent

## Problem Statement

Neon Player contributors cannot use strict mypy to distinguish new mistakes from
existing typing problems. Shared application interfaces describe recording state,
application instances, plugin lookup results, NumPy arrays, and Qt objects too
loosely or incorrectly. Errors at those interfaces produce diagnostics throughout
their callers.

Step 4 of the static testing cleanup addresses these shared contracts first.
Python 3.12 configuration, dependency typing exceptions, and splash UI typing
have already been handled separately. The observed baseline on
`static_testing_updates` is 506 errors in 31 files, across 43 checked source
files. This count describes the current state, not an acceptable error budget.

## Solution

Contributors receive accurate shared types and callers that handle the states
those types permit. Missing recordings, absent plugins, and optional Qt objects
remain explicit. Operations that require a loaded recording use a reference that
has been proved available for that operation.

Preserve playback, plugin behavior, export output, and persisted settings.
Resolve errors in the shared contracts and affected callers without weakening
strict mypy or adding application-wide suppressions. Record unrelated remaining
diagnostics separately so contributors can see what this work establishes.

## User Stories

1. As a contributor, I want the application instance interface to describe its actual lifecycle, so that callers cannot assume an application exists before startup.
2. As a contributor, I want recording-dependent operations to establish recording availability, so that mypy rejects access to an absent recording.
3. As a player user, I want supported actions without a loaded recording to retain their current behavior, so that typing changes do not introduce startup failures.
4. As a contributor, I want recording settings to have an explicit type and lifecycle, so that callers do not access settings before they are available.
5. As a player user, I want loading, unloading, and switching recordings to preserve their behavior, so that state from an earlier recording is not used accidentally.
6. As a contributor, I want a stable recording reference within a synchronous operation, so that repeated property access does not lose established type information.
7. As a plugin author, I want plugin lookup to describe both a found plugin and an absent plugin, so that optional dependencies remain optional.
8. As a plugin author, I want access to a specific plugin's capabilities to establish its concrete type, so that shared plugin interfaces do not require unsafe casts.
9. As a contributor, I want plugin collections and state dictionaries to specify their contents, so that incorrect entries fail type checking.
10. As a contributor, I want shared callbacks and decorators to preserve supported argument and return contracts, so that registered actions remain usable by the property widget framework.
11. As a contributor, I want NumPy annotations to distinguish the dtypes the code relies on, so that numerical operations receive compatible arrays.
12. As a player user, I want gaze and fixation calculations to retain their numerical results, so that type cleanup does not alter visualization or export data.
13. As a contributor, I want timestamps and frame indices to match the values required by existing APIs, so that incompatible arguments fail type checking.
14. As a contributor, I want Qt lookups to preserve concrete widget types or handle missing objects, so that widget methods are called on valid objects.
15. As a player user, I want timeline interactions and playback controls to retain their behavior, so that stricter annotations do not change navigation.
16. As a player user, I want the Export window to retain its start and end timestamps, so that selecting and exporting an interval behaves consistently.
17. As a contributor, I want serialization boundaries to distinguish unstructured input from validated application state, so that arbitrary JSON is not presented as a trusted model.
18. As a maintainer, I want required library types to remain authoritative, so that Neon Player does not duplicate recording or video functionality.
19. As a maintainer, I want before-and-after mypy results for every shared contract change, so that a lower error count cannot hide a weaker check.
20. As a maintainer, I want existing behavior tests to remain green and focused regression coverage where behavior changes, so that type improvements are safe to review.
21. As a contributor, I want unrelated remaining mypy errors to be identified explicitly, so that follow-up work has a reliable starting point.

## Implementation Decisions

- Start from `static_testing_updates` and preserve existing work. Capture a fresh
  mypy baseline before implementation because the branch can change after this
  spec is written.
- Keep Python 3.12, strict mypy, unreachable-code checking, the existing package
  selection, and current dependency-specific exceptions. Do not reduce the
  checked application or disable diagnostic categories to meet acceptance.
- Investigate shared application access, the base plugin interface, recording
  settings, plugin registration and lookup, numerical helpers, and Qt integration.
  Migrate affected consumers together with each contract change.
- Describe application availability truthfully. Distinguish optional access from
  access that requires an initialized player when their runtime contracts differ.
  Do not annotate an optional Qt application instance as an always-present player.
- Make recording and recording-settings availability explicit. Preserve lifecycle
  ordering and existing supported behavior while loading and unloading. Do not
  assume that every callback runs with an active recording.
- Narrow optional recording references once per synchronous operation where
  appropriate. An asynchronous callback must establish availability again rather
  than rely on an earlier check. Introduce a shared loaded-state structure only
  if the observed invariants require it.
- Preserve optional name-based plugin lookup. Establish the concrete plugin type
  before accessing capabilities outside the base plugin interface. Keep persisted
  plugin identifiers and property names compatible.
- Parameterize shared collections, callables, and NumPy arrays with the types
  their consumers require. Follow authoritative library annotations and observed
  values. Avoid introducing elaborate shape types or branded timestamps unless
  a concrete call contract requires them.
- Resolve Qt nullability and overload mismatches at the relevant integration
  point. Prefer existing typed Qt APIs and concrete object references. Do not
  discard fractional values or replace enum members solely to silence mypy
  without checking the intended runtime behavior.
- Preserve descriptor behavior and runtime type introspection used by actions
  and property widgets. A descriptor is not an ordinary callable merely because
  annotating it that way removes errors.
- Keep unstructured JSON and untyped third-party values at explicit boundaries.
  Validate or narrow only the properties needed by the shared contract. Do not
  replace `Any` with an unverified concrete type or an unchecked cast.
- A runtime check is appropriate for a legitimate missing value or external
  boundary. Do not add repetitive guards, broad exception handlers, or silent
  fallbacks to conceal an invalid internal state.
- Use Neon Recording for every recording-data read. Use PL Video for video reads
  outside recordings and all video writes. Inspect the relevant library API
  before changing an integration. Propose a library change if its API prevents a
  correct implementation, and continue independent work while it is unresolved.
- Preserve public plugin behavior, settings and cache formats, export schemas,
  and Export window semantics. Its start may equal its end but cannot follow it.
  This spec does not authorize unrelated behavior changes or format migrations.

## Testing Decisions

- The user confirmed the existing full-project mypy check and existing pytest
  suite as the verification seams. A new application-level harness is not needed.
- Run the normal strict mypy command over the complete configured application.
  Keep the before-and-after diagnostics and map each shared contract change to
  the errors it resolves in the provider and its consumers.
- Acceptance requires no remaining diagnostics caused by the shared contracts
  migrated in this work, no newly introduced application diagnostics, and no new
  application suppressions. Review errors exposed by stronger typing rather than
  assuming every newly visible diagnostic is a runtime regression.
- Full-project mypy is not required to become clean if independent application
  errors remain. Identify those errors and explain why they are outside the
  migrated shared contracts. Do not treat the baseline count as a fixed target.
- Run the existing pytest suite before and after implementation with the virtual
  environment's Qt tools available and the supported headless Qt configuration.
  Ruff lint, formatting, and whitespace checks must also pass.
- Reuse the mock player application and mock Neon Recording fixtures. Existing
  settings tests establish defaults and Export window recovery. Plugin tests
  establish serialized state and gaze, fixation, event, surface, and XDF results.
  Timeline helper tests provide prior art for optional UI objects.
- Add targeted tests when a fix changes runtime behavior or establishes an
  invariant without existing coverage. Assert observable results such as a
  preserved timestamp, export interval, plugin lookup result, or numerical array.
  Do not write tests that merely repeat an annotation or a helper's implementation.
- Cover applicable recording-absent and recording-loaded cases, missing and
  available plugin dependencies, and empty and populated numerical data at the
  existing seams. Exercise lifecycle transitions only where the change relies on
  their ordering. Check action invocation or Qt interaction directly when a
  changed contract affects runtime framework behavior.

## Out of Scope

- Implementing this spec as part of the spec-writing request.
- Fixing every independent mypy diagnostic across the application.
- Expanding third-party dependency exceptions or authoring a complete third-party
  stub package.
- Reworking the completed Python version, Ruff, dependency-import, or splash UI
  changes.
- Restoring the obsolete camera module or reversing its removal.
- Replacing the application framework, introducing a new global state manager, or
  redesigning plugin discovery without evidence that a shared contract needs it.
- Renaming persisted XDF properties, migrating pickle caches, changing dependency
  installation policy, or restructuring the optical-flow algorithm.
- Adding features, changing numerical algorithms, altering export formats, or
  changing playback timing as incidental cleanup.

## Further Notes

This is the shared-application-types step from the conversation, not a request to
start implementation. The spec is ready for an agent to implement using the
confirmed verification seams.

Shared fixes can eliminate many diagnostics, but the effect must be measured on
the actual branch. If a required library lacks an adequate API or type contract,
record the limitation and proposed library change instead of bypassing it in
Neon Player. Leave independent application diagnostics visible for subsequent
work.
