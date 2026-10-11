# Dockable video workspace and eye video panel

Status: ready-for-agent

## Problem Statement

Neon Player currently presents scene video in one central widget. Eye Overlay paints the combined eye video directly into that image. Users can move and resize that overlay, but cannot arrange it as a separate split, tab, or floating window. Other plugins lack a shared way to contribute panels to the video workspace.

Users need flexible video arrangements while preserving the timeline, sidebar, existing scene-video interactions, and export behavior. Rearranging eye video onscreen must not unexpectedly change exported videos, frames, or clipboard images.

## Solution

Introduce the local pyside6-docking package into the video workspace. Keep scene video in a fixed, nonclosable panel, with the timeline and sidebar outside the DockManager. Provide a shared plugin-facing panel API; Eye Overlay is its first user.

The eye video panel starts near the scene video's top-left corner at roughly the existing default size. It supports overlay movement and resizing, split/tab docking, floating, and redocking using standard package controls. Remember one application-wide layout across recordings and application restarts, including placement for absent panels. Closing the eye panel disables its plugin; re-enabling recreates it. Extend “Reset docks” to restore the video workspace defaults.

Retain the existing eye-overlay composition for video exports, frame exports, and clipboard captures independently of panel arrangement. Preserve existing per-recording overlay properties and label their controls as export settings. The eye panel follows shared playback and seeking, clears unsuitable frames, and supports existing playback shortcuts when docked or floating.

## User Stories

1. As a recording analyst, I want a dockable video workspace, so that I can arrange video views for my analysis task.
2. As a recording analyst, I want the timeline outside that workspace, so that rearranging video panels preserves navigation.
3. As a recording analyst, I want the sidebar outside that workspace, so that rearrangement preserves settings access.
4. As a recording analyst, I want a fixed, nonclosable scene panel, so that I retain a stable workspace anchor.
5. As a recording analyst, I want existing scene-video interactions preserved, so that panels do not disrupt analysis.
6. As a Surface Tracking user, I want editing handles and scene-coordinate mapping to remain correct after resizing, so that my edits stay accurate.
7. As an Eye Overlay user, I want enabling the plugin to create an eye video panel, so that I can inspect eye video alongside scene video.
8. As an Eye Overlay user, I want a familiar initial top-left placement and size, so that the default view remains recognizable.
9. As an Eye Overlay user, I want the combined eye-camera image preserved, so that I retain the current eye video content.
10. As an Eye Overlay user, I want to move the overlay using panel controls, so that I can uncover scene-video details.
11. As an Eye Overlay user, I want to resize the panel without distorting video, so that I can inspect eye detail at a useful size.
12. As an Eye Overlay user, I want eye video in a split beside scene video, so that I can see both without overlap.
13. As an Eye Overlay user, I want eye video in a tab group, so that I can use limited workspace space efficiently.
14. As a recording analyst, I want to switch back to the scene tab, so that selecting eye video does not remove scene-video access.
15. As an Eye Overlay user, I want eye video in a floating window, so that I can position it independently.
16. As an Eye Overlay user, I want to redock floating eye video, so that I can return to an overlay, split, or tab arrangement.
17. As an Eye Overlay user, I want closing the panel to disable its plugin, so that the toggle reflects activation consistently.
18. As an Eye Overlay user, I want enabling the plugin again to recreate its panel with remembered placement where possible, so that reopening restores my arrangement.
19. As a recording analyst, I want layout preserved across recording changes, so that I avoid rearranging panels each time.
20. As a recording analyst, I want layout preserved across application restarts, so that I resume my preferred arrangement.
21. As a recording analyst, I want placement remembered when eye video is unavailable, so that a later compatible recording restores it.
22. As a recording analyst, I want restoration to respect disabled plugins, so that a saved layout does not enable unwanted plugins.
23. As a recording analyst, I want a usable fallback for invalid or incompatible layouts, so that saved data does not prevent viewing.
24. As a recording analyst, I want “Reset docks” to reset the video workspace too, so that one action recovers the default arrangement.
25. As a recording analyst, I want reset to leave disabled plugins disabled, so that it does not change plugin activation.
26. As an Eye Overlay user, I want eye video synchronized during playback, so that it corresponds to the scene video.
27. As an Eye Overlay user, I want eye video updated after seeking, so that paused inspection shows the appropriate frame.
28. As an Eye Overlay user, I want an empty state when no sufficiently close frame exists, so that stale eye video is not mistaken for current data.
29. As a recording analyst, I want playback shortcuts with docked eye-panel focus, so that navigation does not require changing focus.
30. As a recording analyst, I want playback shortcuts with floating eye-panel focus, so that floating does not interrupt control.
31. As a recording analyst, I want existing eye-overlay video exports independent of panel placement, so that workspace changes do not alter deliverables.
32. As a recording analyst, I want frame exports and clipboard captures to retain that composition, so that still-image output remains consistent.
33. As an existing Eye Overlay user, I want saved position, scale, opacity, and border properties preserved for exports, so that the integration retains my configuration.
34. As an Eye Overlay user, I want these controls labeled as export settings, so that their effect is clear.
35. As a recording analyst, I want panels and floating windows cleaned up on unload and shutdown, so that stale video views do not remain open.
36. As a plugin author, I want a shared panel API with stable identities and defaults, so that my plugin need not implement workspace persistence.
37. As a plugin author, I want ownership of panel content and lifecycle while Neon owns arrangement, so that responsibilities are clear.
38. As a Neon Player developer, I want the editable sibling docking checkout used during development, so that local changes are immediately available.
39. As a packaged application user, I want the dependency and its assets bundled, so that I need no separate docking installation.

## Implementation Decisions

- Embed DockManager in the main window's existing video presentation slot. Preserve splash/recent-recording transitions and keep outer timeline/sidebar docks intact.
- Retain the existing scene VideoRenderWidget and the main-window reference used by other plugins. Preserve aspect-fit rendering, mouse signals, coordinate mapping, and positioned child widgets used by Surface Tracking.
- Make the scene panel fixed, nonclosable, and nonfloating. Other panels may share its tab group: selecting eye video can temporarily hide scene video without removing it.
- Provide a shared video-workspace registration/removal interface. Each panel has a stable identity, plugin-owned content, default placement, supported features, and an association between user close and its owning plugin.
- Neon owns DockManager, layout persistence, restore, and reset; plugins own content and lifecycle. Eye Overlay is the first client.
- Create one combined grayscale eye video panel. Its initial placement is near the scene video's top-left at approximately the current default size, accounting for panel chrome and independent of saved export settings.
- Support overlay movement/resize, split/tab docking, floating, and redocking. Accept standard tab strips, overlay title bars, and resize frames. Preserve video aspect ratio within the panel.
- Subscribe eye presentation to shared playback-position updates, including seeking. Preserve the freshness rule: frames more than 1/30 second from the requested timestamp are not displayed. Clear stale content and show an empty state for unsuitable or missing frames.
- Preserve existing playback actions and shortcuts with docked or floating eye content focused. Avoid duplicated actions or double execution.
- Separate onscreen eye-panel presentation from export-overlay rendering. Suppress the legacy painted eye overlay in the onscreen scene path to avoid duplicate eye images; retain it for video/frame exports and clipboard capture. Preserve unrelated plugin render layers and headless exports.
- Retain Eye Overlay's existing plugin identity and stored property keys. Position, scale, opacity, border width, and border color remain per-recording export settings, clearly labeled. Panel movement/resizing must not modify them.
- User close disables the plugin through existing activation handling. Re-enabling reconstructs and registers the panel. Distinguish user close from programmatic recording teardown so unloading does not accidentally rewrite intended activation state.
- Queue lifecycle reactions to docking signals. DockManager unregisters panels on close and forbids synchronous manager mutation from callbacks; reopening requires registration rather than showing an unregistered panel.
- Persist one application-wide workspace layout separately from per-recording activation and export properties, using supported JSON-compatible serialization including overlays.
- Retain absent/disabled panel placement without automatically enabling plugins. Reconcile saved data against currently registered identities before restoration because DockManager requires matching identities. New panels use defaults; incompatible data falls back safely. Teardown must not overwrite remembered placement with an empty layout.
- Extend “Reset docks” while preserving its timeline/sidebar behavior. Restore enabled eye video to its default overlay placement without enabling disabled plugins. Reset remembered eye placement too, so reopening does not undo reset.
- Clean up panels, update subscriptions, and floating windows on disable, unload, and shutdown. Remember placement before destruction and prevent rendering from closed recordings.
- Declare and lock the docking dependency with an editable sibling checkout for development. Bundle its code and required assets in standalone builds; runtime must not depend on the checkout.
- A small public initial-overlay-geometry API addition in pyside6-docking is authorized if needed. Prefer a supported package API over private renderer manipulation.

## Testing Decisions

- Primary proposed seam: Qt integration tests at the video-workspace boundary, driven through existing plugin activation, shared playback/seek signals, panel registration, layout save/restore, and reset. Exercise real DockManager behavior and observable widget/plugin state rather than private layout internals. This seam was presented for an optional user expectations check; no additional preference has been recorded yet.
- Good tests assert external behavior: visible content, plugin activation, displayed frames, recovered placement, shortcut execution, and exported output. Avoid implementation-mirroring assertions about helper calls, private widget hierarchies, or private layout trees.
- Reuse pytest-qt, the existing QApplication fixture, mock recordings, and temporary settings storage. The current mock application is minimal and headless; compose or extend a focused integration fixture rather than introduce another production plugin or playback system.
- Cover scene-widget interaction/coordinate mapping after resizing, Surface Tracking compatibility, unchanged outer dock boundaries, and greeting/video transitions.
- Cover activation, close-to-disable, disable/enable, repeated cycles, unload/reload, and shutdown. Process queued events/deferred deletion and verify no duplicate panels, subscriptions, or orphan floating windows.
- Cover default geometry, movement/resize, splits, tabs, float/redock, aspect-fit presentation, updates after transfer, and reset. The docking package's overlay and floating-lifecycle integration suites provide prior art; test Neon's integration contract rather than duplicate its complete docking suite.
- Recreate the workspace from persisted settings and switch recordings with different availability. Cover absent eye video, disabled plugins, new panels, invalid saved data, exact-identity restoration constraints, remembered missing-panel placement, and restoration without activation.
- Use deterministic frames/timestamps to cover playback and seeking, including samples within, at, and beyond the timing tolerance. Verify an unsuitable sample clears previously displayed content.
- Exercise shortcuts with docked and floating focus and assert exactly one action per keypress.
- Supplemental existing seam: the shared application render-to-painter path. Use deterministic scene/eye content to verify output composition remains independent of panel placement while export properties still affect output. Cover video/frame/clipboard callers and headless compatibility without relying solely on codec-byte equality.
- Existing recording-settings tests provide prior art for legacy property loading and corrupted-data fallback. Existing video-export tests provide timestamp-fixture patterns; they do not yet cover panel integration or full rendered composites.
- Perform a native GUI smoke check with actual scene rendering and eye playback through overlay, split, tab, floating, and redocking, plus Surface Tracking interaction and focused-window shortcuts. Mocked/offscreen tests alone cannot establish OpenGL rendering continuity after reparenting. Report the platform/display combinations actually verified.
- Smoke-test a packaged build for dependency imports, assets, workspace creation, and eye presentation without access to the development checkout. Any new package geometry API receives public-contract tests in the docking package.

## Out of Scope

- Moving the timeline or sidebar into the video DockManager.
- Moving, closing, or floating the scene panel itself.
- Exporting panel arrangements or including panel chrome in outputs.
- Separate left/right eye panels; retain the combined eye video.
- Additional concrete plugin panels beyond Eye Overlay.
- Independent eye playback clocks or new playback controls.
- Per-recording workspace layouts, named presets, or multiple saved workspaces.
- Replacing the scene renderer or redesigning Surface Tracking.
- Custom docking chrome, hidden single-panel tab strips, or removal of overlay title bars.
- Broad docking-package changes beyond the small public geometry API addition needed for defaults.

## Further Notes

All sixteen design questions were individually accepted. This spec synthesizes those answers and is published to the local Markdown issue tracker as ready-for-agent. Feature implementation has not been performed as part of spec creation.

Use the glossary terms scene video, eye video, and eye video panel. Keep the plugin name “Eye Overlay” for continuity while referring to its view as the eye video panel regardless of placement.

Follow the recorded ADRs “Keep docking within the video workspace,” “Keep workspace ownership in Neon Player,” and “Separate eye panel placement from export composition.”

The local package already supports overlays, splits, tabs, floating, and layout serialization. It has no special central-anchor API and no public initial-overlay-geometry setter at inspection time. Generic widget transfers preserve identity, but actual Neon OpenGL rendering continuity still needs verification.

The developer package is at /home/marc/pupil_labs/pyside6-docking. Development configuration should express the portable sibling relationship rather than embed this machine-specific absolute path in published metadata.

Exact API names and persistence reconciliation mechanics are implementation choices constrained by the behaviors above. Report native verification evidence accurately rather than infer platform support from mocked/offscreen runs.
