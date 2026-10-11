# Separate eye panel placement from export composition

Eye Overlay currently paints eye video into the shared scene rendering pipeline used for viewing and exports. Its onscreen eye video will move into a dockable panel, while exported videos, frames, and clipboard captures retain the existing overlay composition and per-recording position, scale, opacity, and border settings. Keeping these presentation paths separate prevents workspace rearrangement from changing exported content; the existing property controls will be labeled as export settings.
