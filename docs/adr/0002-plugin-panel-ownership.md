# Keep workspace ownership in Neon Player

Neon Player will provide a shared API through which plugins register and remove video workspace panels, with Eye Overlay as the first user. Neon owns the workspace and layout persistence, while plugins own their panel content and lifecycle, so each plugin does not have to manage shared layout state independently.
