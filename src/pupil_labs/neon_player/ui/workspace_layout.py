"""Reconcile public docking JSON without creating unavailable plugin widgets."""

from copy import deepcopy
import math
from typing import Any

Layout = dict[str, Any]


def _tree_ids(node: Layout | None, depth: int = 0) -> set[str]:
    if node is None:
        return set()
    if depth > 64:
        raise ValueError("Workspace tree is too deep")
    if node["type"] == "tabs":
        docks = node["docks"]
        if (
            not isinstance(docks, list)
            or not docks
            or any(not isinstance(d, str) for d in docks)
        ):
            raise ValueError("Invalid workspace tabs")
        return set(docks)
    if node["type"] != "split":
        raise ValueError("Unknown workspace node")
    return set().union(*(_tree_ids(child, depth + 1) for child in node["children"]))


def layout_ids(layout: Layout) -> set[str]:
    version = layout["schema_version"]
    if type(version) is not int or version not in (1, 2):
        raise ValueError("Unsupported workspace schema")
    fields = {"schema_version", "main", "floating", "hidden"}
    if version == 2:
        fields.add("overlays")
    if set(layout) != fields:
        raise ValueError("Invalid workspace fields")
    ids = _tree_ids(layout["main"])
    for floating in layout["floating"]:
        ids.update(_tree_ids(floating["root"]))
    ids.update(entry["dock_id"] for entry in layout["hidden"])
    ids.update(entry["dock_id"] for entry in layout.get("overlays", []))
    return ids


def _project_tree(node: Layout | None, ids: set[str], depth: int = 0) -> Layout | None:
    if node is None:
        return None
    if depth > 64:
        raise ValueError("Workspace tree is too deep")
    result = deepcopy(node)
    if node["type"] == "tabs":
        docks = [dock for dock in node["docks"] if dock in ids]
        if not docks:
            return None
        result["docks"] = docks
        # An active tab removed by projection is normal; an invalid active ID is
        # left intact so the package's strict restore rejects the document.
        if node["active"] in node["docks"] and node["active"] not in docks:
            result["active"] = docks[0]
        return result
    children, sizes = node["children"], node["sizes"]
    if len(children) < 2 or len(children) != len(sizes):
        raise ValueError("Invalid workspace split")
    if any(
        type(s) not in (int, float) or not math.isfinite(s) or s <= 0 for s in sizes
    ):
        raise ValueError("Invalid workspace sizes")
    if not math.isclose(sum(sizes), 1.0):
        raise ValueError("Unnormalized workspace sizes")
    kept = [
        (_project_tree(child, ids, depth + 1), size)
        for child, size in zip(children, sizes)
    ]
    kept = [(child, size) for child, size in kept if child is not None]
    if not kept:
        return None
    if len(kept) == 1:
        return kept[0][0]
    result["children"] = [child for child, _ in kept]
    total = sum(size for _, size in kept)
    result["sizes"] = [size / total for _, size in kept]
    return result


def _project(
    layout: Layout, ids: set[str], overlay_parents: set[str] | None = None
) -> Layout:
    layout_ids(layout)
    result = deepcopy(layout)
    result["schema_version"] = 2
    result["main"] = _project_tree(layout["main"], ids)
    result["floating"] = []
    for container in layout["floating"]:
        root = _project_tree(container["root"], ids)
        if root is not None:
            result["floating"].append({**deepcopy(container), "root": root})
    result["hidden"] = [
        deepcopy(entry) for entry in layout["hidden"] if entry["dock_id"] in ids
    ]
    for entry in result["hidden"]:
        entry["placement"].setdefault("overlay", None)
    result["overlays"] = [
        deepcopy(entry)
        for entry in layout.get("overlays", [])
        if entry["dock_id"] in ids
        and entry["parent_id"] in (ids | (overlay_parents or set()))
    ]
    return result


def _graft(template: Layout, area: Layout, anchor: str) -> Layout:
    """Insert a remembered fragment around its surviving tab area."""
    if template["type"] == "tabs":
        if anchor in template["docks"]:
            before, after = (
                template["docks"][: template["docks"].index(anchor)],
                template["docks"][template["docks"].index(anchor) + 1 :],
            )
            index = area["docks"].index(anchor)
            area["docks"][index:index] = before
            index = area["docks"].index(anchor) + 1
            area["docks"][index:index] = after
            if template["active"] in before + after:
                area["active"] = template["active"]
            return area
    else:
        template["children"] = [
            _graft(child, area, anchor) for child in template["children"]
        ]
    return template


def _merge_tree(live: Layout, defaults: Layout | None, missing: set[str]) -> Layout:
    """Fill absent membership at its nearest surviving ancestor/neighbor."""
    if defaults is None:
        return live
    surviving = _tree_ids(live)
    remaining = missing & _tree_ids(defaults)

    def replace_area(node: Layout, anchor: str, fragment: Layout) -> Layout:
        if node["type"] == "tabs":
            return _graft(fragment, node, anchor) if anchor in node["docks"] else node
        node["children"] = [
            replace_area(child, anchor, fragment) for child in node["children"]
        ]
        return node

    def visit(node: Layout) -> None:
        nonlocal live
        if node["type"] == "split":
            for child in node["children"]:
                visit(child)
        members = _tree_ids(node)
        absent = remaining & members
        anchors = surviving & members
        if absent and anchors:
            anchor = sorted(anchors)[0]
            fragment = _project_tree(node, absent | {anchor})
            assert fragment is not None
            live = replace_area(live, anchor, fragment)
            remaining.difference_update(absent)

    visit(defaults)
    return live


def reconcile_layout(preferred: Layout, defaults: Layout, ids: set[str]) -> Layout:
    """Project to exact identities, filling missing identities from defaults.

    Also used in reverse when saving: live placement takes precedence while the
    previous preference supplies panels absent from the current recording.
    The DockManager remains responsible for final schema/feature validation.
    """
    result = _project(preferred, ids)
    if "scene-video" not in _tree_ids(result["main"]):
        raise ValueError("Workspace must keep the scene in the main container")
    missing = ids - layout_ids(result)
    if missing:
        extra = _project(defaults, missing | {"scene-video"}, layout_ids(result))
        result["main"] = _merge_tree(result["main"], defaults["main"], missing)
        existing = {container["id"]: container for container in result["floating"]}
        for container in extra["floating"]:
            current = existing.get(container["id"])
            if current is not None:
                original = next(
                    item
                    for item in defaults["floating"]
                    if item["id"] == container["id"]
                )
                if _tree_ids(current["root"]) & _tree_ids(original["root"]):
                    current["root"] = _merge_tree(
                        current["root"], original["root"], missing
                    )
                    continue
            while container["id"] in existing:
                container["id"] = "remembered-" + container["id"]
            result["floating"].append(container)
            existing[container["id"]] = container
        result["hidden"].extend(extra["hidden"])
        result["overlays"].extend(extra["overlays"])
    return result
