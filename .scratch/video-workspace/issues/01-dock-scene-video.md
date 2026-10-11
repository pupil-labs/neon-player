# 01: Dock the scene video

**What to build:** A video workspace backed by the local pyside6-docking package, with the existing scene video inside a fixed, nonclosable panel. Users retain the timeline, sidebar, greeting screens, and scene-video analysis interactions while the application gains a workspace that can host additional panels.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [x] Declare and lock pyside6-docking using the editable sibling checkout for development, without embedding a machine-specific absolute location in published metadata.
- [x] Embed DockManager only in the existing video presentation slot; timeline and sidebar remain outside it.
- [x] Keep the scene panel fixed, nonclosable, and nonfloating, using standard package chrome.
- [x] Retain the existing scene VideoRenderWidget and the reference used by plugins, including its aspect-fit rendering, mouse signals, coordinate mapping, and positioned child widgets.
- [x] Preserve splash/recent-recording transitions and recording load/unload behavior.
- [x] Existing scene rendering and unrelated plugin render layers remain intact, including the legacy Eye Overlay behavior until its replacement in ticket 02.
- [x] Verify scene playback, resize behavior, and Surface Tracking editing handles and coordinate interactions in the new workspace.
- [x] Include Qt behavioral checks at the workspace boundary and a native scene-rendering smoke check; report the platform/display combination verified.

## Comments

Implemented on 2026-10-11. The scene video is embedded in a fixed, nonclosable,
nonfloating panel; the existing widget and plugin rendering remain intact.
Application shutdown unloads recording data before disposing the workspace.

Validation:

- Native Linux/X11 (xcb), Python 3.12, PySide6 6.11.0: real application navigation,
  displayed scene frame updates, legacy eye overlay, scene resizing, Surface
  Tracking handle placement/drag, recording close, and shutdown checks passed.
- Full suite through the locked uv environment: 85 passed, including doctests
  and the native workspace check.
- Typechecking: 394 existing errors in the synced environment; unchanged source
  checked with the same dependencies produces the same diagnostic signatures.
  No new typechecking errors were introduced.
- Standards and ticket-compliance reviews: no actionable findings on either axis.
- Other native platforms/display backends were not verified by this ticket.

All acceptance criteria are complete. The triage status vocabulary is retained;
completion is recorded here and in the checked acceptance criteria.
