# 02: Introduce the eye video panel

**What to build:** Enabling Eye Overlay creates a synchronized eye video panel at a familiar default overlay position. Users can move and resize its presentation while existing exported eye-overlay composition and saved export settings remain independent. Plugins gain a shared panel API, and closing the eye panel disables its plugin consistently.

**Blocked by:** 01 — Dock the scene video.

**Status:** ready-for-agent

- [ ] Provide a shared registration/removal interface with stable panel identities, plugin-owned content, default placement, supported features, and user-close association with the owning plugin.
- [ ] Neon owns the workspace and layout coordination; plugins own panel content and lifecycle. Eye Overlay uses the shared API rather than managing the DockManager directly.
- [ ] Preserve the existing Eye Overlay plugin identity and show one combined grayscale eye video panel.
- [ ] Initially place it near the scene video's top-left corner at roughly the existing default size, allowing for standard title-bar and resize-frame chrome. Default panel geometry is independent of per-recording export properties.
- [ ] Add a small public initial-overlay-geometry API to pyside6-docking if necessary, with public-contract tests; avoid relying on private renderer internals.
- [ ] Panel movement and resize preserve video aspect ratio and do not alter export properties.
- [ ] Follow the existing shared playback/seek position. Frames more than 1/30 second away are not displayed; missing or unsuitable frames clear stale content and show an empty state.
- [ ] Present eye video only in the panel onscreen, while retaining the existing painted composition for video export, frame export, and clipboard capture. Preserve unrelated render layers and headless exports.
- [ ] Preserve existing stored position, scale, opacity, border width, and border color values and property keys; label their controls as export settings.
- [ ] Closing the panel disables Eye Overlay through the existing activation flow; enabling it again reconstructs and registers one panel. Persistent placement is added in ticket 04.
- [ ] Queue close-related lifecycle reactions so callbacks do not synchronously mutate DockManager; programmatic teardown does not rewrite intended per-recording activation state.
- [ ] Include workspace integration checks for activation, close/reopen, timing boundaries, empty states, and default geometry, plus deterministic render-to-painter checks that panel movement does not change exported composition.
