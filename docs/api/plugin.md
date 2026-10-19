!!! info "Plugin API"
The classes and methods listed below form the public API, which is recommended
for extending the functionality of Neon Player. Since Neon Player uses PySide6
under the hood, a plugin can technically reach other application and Qt
components too. However, those undocumented parts may change without notice and
are not guaranteed to be compatible with the future releases of Neon Player.

    If the currently available methods are not sufficient for your use case, feel
    free to [open an issue](https://github.com/pupil-labs/neon-player/issues) in
    the GitHub repository.

::: pupil_labs.neon_player.Plugin
options:
inherited_members: false
members: - app - job_manager - headless - recording - render_layer - get_timeline - global_properties - on_recording_loaded - on_disabled - render - trigger_scene_update - get_scene_idx_for_time - is_time_gray - get_cache_path - load_cached_json - save_cached_json - add_dynamic_action - register_action - unregister_action - register_timeline_action - unregister_timeline_action - register_data_point_action - unregister_data_point_action - user_confirm
