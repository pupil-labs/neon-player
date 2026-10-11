# 03: Support floating eye video and playback controls

**What to build:** Users can arrange eye video as an overlay, split, tab, or floating window and return it to the workspace without interrupting video updates or playback controls. Panel and window ownership remains correct across plugin disable, recording changes, and application shutdown.

**Blocked by:** 02 — Introduce the eye video panel.

**Status:** ready-for-agent

- [ ] Enable and verify overlay movement/resize, split/tab arrangements, floating, and redocking through standard package controls.
- [ ] Selecting the eye tab may temporarily hide scene video; switching back restores access to the fixed, nonclosable scene panel.
- [ ] Eye video continues updating through placement changes and preserves aspect-fit presentation; no duplicate or stale legacy eye image appears in the scene video.
- [ ] Existing playback shortcuts work with both docked and floating eye-panel focus, executing exactly once per keypress.
- [ ] Closing a floating eye panel follows the same close-to-disable behavior as a docked panel, and enabling the plugin recreates it safely.
- [ ] Disable, recording unload, and application shutdown remove panels, disconnect update subscriptions, and retire floating windows. No rendering reads a closed recording.
- [ ] Repeated enable/disable, float/redock, and recording-change cycles leave no duplicate panels, subscriptions, orphan windows, or destroyed live content.
- [ ] Include Qt integration checks for observable placement, focus, activation, and cleanup behavior, processing queued events and deferred deletion.
- [ ] Verify actual scene rendering and eye playback in a native GUI through overlay, split, tab, floating, and redocking, including Surface Tracking and focused-window shortcuts. Report tested platform/display combinations and verification limits.
- [ ] Retain the export-composition guarantees from ticket 02 across all arrangements.
