import logging
import pandas as pd

from pupil_labs.neon_player.plugins.events.event_type import EventType, IMMUTABLE_EVENTS
from pupil_labs.neon_recording import NeonRecording


def _load_events_from_recording(
    recording: NeonRecording, global_event_types: list[str] = []
) -> tuple[list[EventType], dict[str, list[int]]]:
    """
    Loads events from the recording and additionally creates event types that
    are defined in global settings.

    Returns all created event types and the events as {event name: list of timestamps}.
    """
    event_type_cache: dict[str, EventType] = {}
    events: dict[str, list[int]] = {}

    for event in recording.events:
        event_name = str(event.event)

        # Look up or create the event type
        et = event_type_cache.get(event_name, None)
        if et is None:
            et = EventType.from_name(event_name)
            event_type_cache[event_name] = et

        # Add event to the dictionary
        if et.name not in events:
            events[et.name] = []
        events[et.name].append(int(event.time))

    for event_name in global_event_types:
        if event_name in event_type_cache:
            continue

        event_type_cache[event_name] = EventType.from_name(event_name)

    return list(event_type_cache.values()), events


def _load_events_from_cache(
    cached_events: dict, known_event_types: list[EventType]
) -> tuple[list[EventType], dict]:
    """
    Load event data from a cache stored in events.json. All event types are expected
    to either be immutable (recording.begin, recording.end) or have been previously
    defined and stored in the known_event_types list.

    Returns event types that are present in the cache as well as events themselves.
    """
    known_event_types_by_uid = {et.uid: et for et in known_event_types}
    for event_name in IMMUTABLE_EVENTS:
        et = EventType.from_name(event_name)
        known_event_types_by_uid[et.uid] = et

    event_type_cache = {}
    events = {}
    for uid in cached_events:
        if uid in event_type_cache:
            continue

        if uid not in known_event_types_by_uid:
            raise ValueError(f"Event type with uid {uid} not found")

        et = known_event_types_by_uid[uid]
        event_type_cache[uid] = et
        events[et.name] = cached_events[uid]

    return list(event_type_cache.values()), events


def _load_events_from_dataframe(events_df: pd.DataFrame) -> dict[str, list[int]]:
    events = {}
    for name, group in events_df.groupby("name"):
        name = str(name)
        if name in IMMUTABLE_EVENTS:
            logging.warning(f"Skipping immutable event '{name}' from imported CSV")
            continue

        events[name] = group["timestamp [ns]"].astype(int).tolist()

    return events
