Plugin initialization:

- Plugin constructor is executed
- Saved state from `settings.json` in recording cache is applied
- The [`on_recording_loaded`][pupil_labs.neon_player.Plugin.on_recording_loaded] method is executed
- Plugin settings are added to the sidebar

Rendering:

- Override `render`

Plugin deactivation:

- [`on_disabled`][pupil_labs.neon_player.Plugin.on_disabled]
