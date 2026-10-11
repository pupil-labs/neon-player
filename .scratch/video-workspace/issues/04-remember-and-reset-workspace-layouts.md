# 04: Remember and reset workspace layouts

**What to build:** Users retain one preferred video-workspace arrangement across recordings and application restarts, including when a plugin panel is temporarily unavailable. They can recover the default arrangement with “Reset docks” without changing which plugins are enabled.

**Blocked by:** 03 — Support floating eye video and playback controls.

**Status:** ready-for-agent

- [ ] Store one application-wide workspace layout using supported JSON-compatible docking serialization, including overlays, split/tab placement, and floating placement/size.
- [ ] Keep workspace state separate from existing per-recording plugin activation and export properties.
- [ ] Save and restore layout across application restart and recording changes; teardown must not overwrite remembered placement with an empty workspace.
- [ ] Remember placement for absent or disabled panels, including recordings without eye video; restoration never automatically enables their plugins.
- [ ] Re-enabling Eye Overlay or returning to a recording with eye video restores remembered placement where possible.
- [ ] Reconcile saved data with currently registered stable identities before calling package restoration, which requires matching identities. Newly introduced panels use their defaults.
- [ ] Invalid, incompatible, or unrestorable layout data falls back to usable defaults without preventing recording viewing or losing unrelated per-recording settings.
- [ ] Extend “Reset docks” to restore the scene panel and return enabled eye video to default overlay placement, retaining the existing timeline/sidebar reset behavior.
- [ ] Reset also updates remembered placement for absent panels so later reopening does not undo reset; disabled plugins stay disabled.
- [ ] Include behavioral round-trip checks using isolated application settings and recreated workspaces, covering overlay/split/tab/floating arrangements, disabled/missing/new panels, recording changes, invalid data, close/reopen, and reset.
- [ ] Verify restored arrangements keep playback, shortcuts, and export composition working through the integration boundaries already established.
