Key components:

- Standard Python class
- Qt Framework is used for application
- API endpoints are described in the reference, methods that are not described here should not
  be considered part of the public API, and backward compatibility might not be guaranteed for
  such methods. If you lack some functionality for implementing a plugin, feel free to describe
  it in an issue on GitHub.
- `@property` - plugin settings that need to be exposed in the UI and/or saved between sessions
- `@action` - clickable buttons to launch certain actions
- Cache
- Can interact with other components of Neon Player
  - Main application
  - Timeline
  - Job manager
