# 03: Support floating eye video and playback controls

**What to build:** Users can arrange eye video as an overlay, split, tab, or floating window and return it to the workspace without interrupting video updates or playback controls. Panel and window ownership remains correct across plugin disable, recording changes, and application shutdown.

**Blocked by:** 02 — Introduce the eye video panel.

**Status:** ready-for-agent

- [x] Enable and verify overlay movement/resize, split/tab arrangements, floating, and redocking through standard package controls.
- [x] Selecting the eye tab may temporarily hide scene video; switching back restores access to the fixed, nonclosable scene panel.
- [x] Eye video continues updating through placement changes and preserves aspect-fit presentation; no duplicate or stale legacy eye image appears in the scene video.
- [x] Existing playback shortcuts work with both docked and floating eye-panel focus, executing exactly once per keypress.
- [x] Closing a floating eye panel follows the same close-to-disable behavior as a docked panel, and enabling the plugin recreates it safely.
- [x] Disable, recording unload, and application shutdown remove panels, disconnect update subscriptions, and retire floating windows. No rendering reads a closed recording.
- [x] Repeated enable/disable, float/redock, and recording-change cycles leave no duplicate panels, subscriptions, orphan windows, or destroyed live content.
- [x] Include Qt integration checks for observable placement, focus, activation, and cleanup behavior, processing queued events and deferred deletion.
- [x] Verify actual scene rendering and eye playback in a native GUI through overlay, split, tab, floating, and redocking, including Surface Tracking and focused-window shortcuts. Report tested platform/display combinations and verification limits.
- [x] Retain the export-composition guarantees from ticket 02 across all arrangements.

## Comments

Implemented floating support by enabling Eye Overlay's FLOATABLE feature. Standard Pop out, overlay movement/resize, split/tab, floating, and redocking controls reuse the same eye content widget. Playback shortcut context is assigned in the shared Playback action registration path, including actions contributed later by plugins; the existing actions execute once with docked, floating, and redocked focus.

The existing queued close and ownership flow handles floating-window close, disable, unload, and shutdown without extra lifecycle machinery. Integration checks cover repeated close/reopen, activation preservation during programmatic teardown, retired floating windows after deferred deletion, distinct recording replacement, and guarded closed media. The test media boundary now supports frame stepping and rejects logged Qt-slot exceptions.

Validation: 96 tests passed. Four new native real-application checks cover the actual Pop out menu, all built-in playback shortcuts and later registered Playback actions, lifecycle cycles, actual OpenGL scene pixels and changing eye frames through all arrangements, tab return to scene, overlay controls, Surface Tracking handle coordinates/dragging, and independent exported composition. Native screen captures raise the main window and park the cursor off video to exclude occlusion. Verified on Linux X11 with Qt 6.11.0 and synthetic QTest input; other platforms and physical-input qualification were not exercised. Mypy retains 382 pre-existing errors, with zero new diagnostics relative to ticket 02.

Standards review: no findings. Spec review identified plugin-added Playback shortcuts; policy moved to shared action registration and re-reviewed with no remaining findings. Persistent layout and reset remain ticket 04.
