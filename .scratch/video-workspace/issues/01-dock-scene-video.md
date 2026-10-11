# 01: Dock the scene video

**What to build:** A video workspace backed by the local pyside6-docking package, with the existing scene video inside a fixed, nonclosable panel. Users retain the timeline, sidebar, greeting screens, and scene-video analysis interactions while the application gains a workspace that can host additional panels.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] Declare and lock pyside6-docking using the editable sibling checkout for development, without embedding a machine-specific absolute location in published metadata.
- [ ] Embed DockManager only in the existing video presentation slot; timeline and sidebar remain outside it.
- [ ] Keep the scene panel fixed, nonclosable, and nonfloating, using standard package chrome.
- [ ] Retain the existing scene VideoRenderWidget and the reference used by plugins, including its aspect-fit rendering, mouse signals, coordinate mapping, and positioned child widgets.
- [ ] Preserve splash/recent-recording transitions and recording load/unload behavior.
- [ ] Existing scene rendering and unrelated plugin render layers remain intact, including the legacy Eye Overlay behavior until its replacement in ticket 02.
- [ ] Verify scene playback, resize behavior, and Surface Tracking editing handles and coordinate interactions in the new workspace.
- [ ] Include Qt behavioral checks at the workspace boundary and a native scene-rendering smoke check; report the platform/display combination verified.
