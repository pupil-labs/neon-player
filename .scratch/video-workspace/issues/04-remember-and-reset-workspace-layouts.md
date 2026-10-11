# 04: Remember and reset workspace layouts

**What to build:** Users retain one preferred video-workspace arrangement across recordings and application restarts, including when a plugin panel is temporarily unavailable. They can recover the default arrangement with “Reset docks” without changing which plugins are enabled.

**Blocked by:** 03 — Support floating eye video and playback controls.

**Status:** ready-for-agent

- [x] Store one application-wide workspace layout using supported JSON-compatible docking serialization, including overlays, split/tab placement, and floating placement/size.
- [x] Keep workspace state separate from existing per-recording plugin activation and export properties.
- [x] Save and restore layout across application restart and recording changes; teardown must not overwrite remembered placement with an empty workspace.
- [x] Remember placement for absent or disabled panels, including recordings without eye video; restoration never automatically enables their plugins.
- [x] Re-enabling Eye Overlay or returning to a recording with eye video restores remembered placement where possible.
- [x] Reconcile saved data with currently registered stable identities before calling package restoration, which requires matching identities. Newly introduced panels use their defaults.
- [x] Invalid, incompatible, or unrestorable layout data falls back to usable defaults without preventing recording viewing or losing unrelated per-recording settings.
- [x] Extend “Reset docks” to restore the scene panel and return enabled eye video to default overlay placement, retaining the existing timeline/sidebar reset behavior.
- [x] Reset also updates remembered placement for absent panels so later reopening does not undo reset; disabled plugins stay disabled.
- [x] Include behavioral round-trip checks using isolated application settings and recreated workspaces, covering overlay/split/tab/floating arrangements, disabled/missing/new panels, recording changes, invalid data, close/reopen, and reset.
- [x] Verify restored arrangements keep playback, shortcuts, and export composition working through the integration boundaries already established.

## Comments

Implemented application-wide video workspace preferences using the docking package's public JSON serialization. The workspace reconciles stable panel identities before restoration and retains placement for unavailable or disabled panels without activating plugins. Recording teardown preserves the remembered layout; newly registered panels use their defaults. Overlay anchors and shared floating tab groups survive temporary panel removal.

“Reset docks” resets registered panels and discards preferences for absent panels while preserving plugin activation and the existing timeline/sidebar reset. Workspace placement remains separate from per-recording export properties. Malformed, incompatible, and rejected layouts recover to usable defaults without discarding unrelated settings.

Behavioral tests use isolated settings and recreated application processes for overlay, split, tab, and floating layouts, recording changes, disabled/missing/new panels, malformed data, close/reopen, reset, playback shortcuts, and export composition. Native Qt coverage runs on Linux X11 with Qt 6.11.0.

Validation: 109 tests passed in 76.23 seconds. Type checking reports the existing 382 errors in 29 files, with no new diagnostics compared with Ticket 03. Standards and Spec reviews have zero remaining findings; both reconciliation findings from the Spec review were corrected and covered by regression tests.
