# 05: Deliver the workspace in standalone builds

**What to build:** A standalone Neon Player build includes the docking dependency and required assets and runs the complete scene/eye video workspace without access to the developer checkout. Users receive the same panel, playback, persistence, reset, and export behavior as the development application.

**Blocked by:** 04 — Remember and reset workspace layouts.

**Status:** ready-for-agent

- [ ] Bundle pyside6-docking code and all required assets through the existing standalone build system, including any geometry API addition required by ticket 02.
- [ ] Document how the build obtains the agreed dependency revision; preserve the editable sibling workflow for development without making end-user runtime depend on that checkout.
- [ ] A packaged application launches, loads package assets, displays the fixed scene panel, and opens the eye video panel without importing from the developer checkout.
- [ ] Perform a packaged native smoke check covering playback/seeking, missing-frame clearing, overlay/split/tab/floating arrangements, redocking, and focused-panel playback shortcuts.
- [ ] Exercise panel close/re-enable, recording unload/reload, restart restoration, unavailable eye video, and “Reset docks” in the packaged workflow.
- [ ] Verify timeline/sidebar and Surface Tracking interactions remain usable, and floating windows retire on unload/shutdown.
- [ ] Verify exported videos, frames, and clipboard captures retain the independent eye-overlay composition and legacy saved export properties in the packaged application, including headless export compatibility.
- [ ] Run the targeted behavioral regression checks introduced by prior tickets and the relevant existing settings/export checks; fix packaging-specific regressions.
- [ ] Record the package revision, build configuration, native platform/display combinations exercised, outcomes, and any remaining platform-specific verification limits. Do not infer native support from mocked or offscreen tests.
