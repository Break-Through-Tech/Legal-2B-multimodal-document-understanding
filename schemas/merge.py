"""Merge per-document JSON schemas into a single schema per document class.

Why this exists
---------------
Every document in the OmniAI benchmark ships its own ``json_schema``, and documents of the
same class often ask for different fields. In the real pipeline a new document arrives with
no schema: the classifier predicts its class, and extraction needs one schema for that class.

The merge rules, in plain words
-------------------------------
- Fields: keep every field that any training document has.
- Required: a field is required only if every document requires it.
- Types and allowed values (enums) are combined: a field that is a "string" in one document
  and a "number" in another accepts both. "integer" folds into "number".
- Other constraints (format, pattern, not) are kept only if every document agrees on them.
- Objects are closed (``additionalProperties: false``): the class schema lists every field
  seen in training, so any other field is unexpected, for example one the extractor invented.

Together these rules guarantee: if one of the input schemas accepts a document, and that
document only uses fields the input schema declares, then the merged schema accepts it too.
"""

import copy
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

# A JSON schema is just a dictionary, e.g. {"type": "string", "description": "..."}.
Schema = dict[str, Any]

# The keywords the benchmark's schemas use. Anything else makes the merge stop with an error
# instead of silently ignoring it, because an unfamiliar keyword could change what's valid.
SUPPORTED_KEYWORDS = {
    "type", "title", "description", "enum", "format", "pattern", "not",
    "properties", "required", "additionalProperties", "items",
}
VALID_TYPES = {"object", "array", "string", "number", "integer", "boolean", "null"}

# Constraints that survive the merge only when every document has the same value.
AGREED_KEYWORDS = ["format", "pattern", "not"]

# How the benchmark marks a field that must never be null.
NOT_NULL = {"type": "null"}


@dataclass(frozen=True)
class Conflict:
    """A place where the input schemas disagreed and the merge had to make a choice."""

    path: str  # where in the schema, e.g. "$.transactions[].type"
    keyword: str  # which rule was involved, e.g. "enum"
    detail: str  # what the merge did about it


def merge_schemas(schemas: list[Schema]) -> tuple[Schema, list[Conflict]]:
    """Merge several documents' schemas into one class schema (rules in the module docstring).

    Fields appear in the order they are first seen, so pass the schemas in a stable order
    (for example sorted by document id) to get the same output every run.

    Returns the merged schema plus a list of every conflict that was resolved, so the
    choices can be reviewed instead of hidden.
    """
    if not schemas:
        raise ValueError("need at least one schema to merge")
    conflicts: list[Conflict] = []
    merged = _merge_nodes(schemas, "$", conflicts)
    # The merged schema may still point at small pieces of the inputs (like enum values).
    # Copying it means callers can edit the result without changing the originals.
    return copy.deepcopy(merged), conflicts


def allow_null(schema: Schema) -> Schema:
    """Return a copy of ``schema`` where every field also accepts null.

    The benchmark's ground truth uses null for a field that is blank on the page (some
    descriptions even say "return null if not signed"), but its schemas never allow null,
    so many ground truths fail their own schema. Allowing null matches how the data is
    labeled, and it lets the extractor answer "blank" instead of inventing a value.

    Left unchanged:
    - fields the benchmark marks as never-null (``"not": {"type": "null"}``)
    - the root object and the entries of a list: null means "blank field", not
      "missing document" or "missing list entry"
    """
    schema = copy.deepcopy(schema)  # work on a copy so the caller's schema is untouched
    _make_fields_nullable(schema)
    return schema


# --- Merging, one schema node at a time ------------------------------------------------


def _merge_nodes(nodes: list[Schema], path: str, conflicts: list[Conflict]) -> Schema:
    """Merge the schema nodes that sit at the same path in different documents."""
    for node in nodes:
        _check_supported(node, path)

    merged: Schema = {"type": _merge_types(nodes, path, conflicts)}

    # Text annotations: keep the wording most documents use.
    for key in ["title", "description"]:
        texts = [node[key] for node in nodes if key in node]
        if texts:
            merged[key] = _most_common(texts)

    enum = _merge_enums(nodes, path, conflicts)
    if enum is not None:
        merged["enum"] = enum

    for key in AGREED_KEYWORDS:
        value = _agreed_value(nodes, key, path, conflicts)
        if value is not None:
            merged[key] = value

    # "required" only has an effect on objects. Note it when a document put it somewhere else
    # (PROXY_VOTING puts one on a list), since it's dropped here.
    if any("required" in node and "properties" not in node for node in nodes):
        conflicts.append(Conflict(path, "required", "ignored: 'required' only applies to objects"))

    # Objects: merge their fields, then close them to fields seen in training.
    object_nodes = [node for node in nodes if "properties" in node]
    if object_nodes:
        merged["properties"] = _merge_properties(object_nodes, path, conflicts)
        required = _merge_required(object_nodes, merged["properties"], path, conflicts)
        if required:
            merged["required"] = required
        merged["additionalProperties"] = False

    # Arrays: merge the schema of their entries.
    item_nodes = [node["items"] for node in nodes if "items" in node]
    if item_nodes:
        merged["items"] = _merge_nodes(item_nodes, path + "[]", conflicts)

    return merged


def _check_supported(node: Schema, path: str) -> None:
    """Stop with a clear error on schema content this module can't merge safely."""
    if not isinstance(node, dict):
        raise TypeError(f"{path}: expected a schema object, got {node!r}")

    unknown = set(node) - SUPPORTED_KEYWORDS
    if unknown:
        raise ValueError(f"{path}: unsupported keyword(s) {sorted(unknown)}")

    if "type" not in node:
        raise ValueError(f"{path}: every field needs a 'type'")
    invalid = _types_of(node) - VALID_TYPES
    if invalid:
        raise ValueError(f"{path}: invalid type(s) {sorted(invalid)}")

    # Objects must list their fields and arrays must describe their entries. Otherwise the
    # merge can't tell "this object has no fields" apart from "any fields are fine".
    if "object" in _types_of(node) and "properties" not in node:
        raise ValueError(f"{path}: object without 'properties'")
    if "array" in _types_of(node) and "items" not in node:
        raise ValueError(f"{path}: array without 'items'")

    # Class schemas are always closed, so only a simple true/false can be replaced safely.
    if "additionalProperties" in node and not isinstance(node["additionalProperties"], bool):
        raise TypeError(f"{path}: only true/false is supported for 'additionalProperties'")


def _types_of(node: Schema) -> set[str]:
    """A node's declared type(s) as a set, e.g. {"string"} or {"string", "null"}."""
    declared = node["type"]
    if isinstance(declared, str):
        return {declared}
    return set(declared)


def _merge_types(nodes: list[Schema], path: str, conflicts: list[Conflict]) -> str | list[str]:
    """Combine every declared type. One type stays a string; several become a list."""
    types: set[str] = set()
    for node in nodes:
        types |= _types_of(node)

    if "integer" in types and "number" in types:
        types.discard("integer")  # every integer is already a number

    type_list = sorted(types)
    if len(type_list) == 1:
        return type_list[0]
    conflicts.append(Conflict(path, "type", f"widened to {type_list}"))
    return type_list


def _merge_enums(nodes: list[Schema], path: str, conflicts: list[Conflict]) -> list[Any] | None:
    """Combine the allowed values, or return None (no enum) if any document allows any value."""
    enums = [node.get("enum") for node in nodes]
    if all(enum is None for enum in enums):
        return None
    if any(enum is None for enum in enums):
        conflicts.append(Conflict(path, "enum", "dropped: some documents allow any value"))
        return None

    # Collect values in the order they first appear. Values are compared by their JSON text,
    # so True and 1 (equal in Python) count as different values, just like in JSON.
    combined: list[Any] = []
    seen: set[str] = set()
    for enum in enums:
        for value in enum:
            key = json.dumps(value)
            if key not in seen:
                seen.add(key)
                combined.append(value)

    # Only report a conflict when documents allow different values, not just a different order.
    first_values = {json.dumps(value) for value in enums[0]}
    for enum in enums[1:]:
        if {json.dumps(value) for value in enum} != first_values:
            conflicts.append(Conflict(path, "enum", f"combined into {len(combined)} values"))
            break
    return combined


def _agreed_value(nodes: list[Schema], key: str, path: str, conflicts: list[Conflict]) -> Any:
    """Return the value of ``key`` if every node has the same one; otherwise return None.

    None means "leave this constraint out". Leaving out a constraint that some documents
    lack (or disagree on) makes the schema more permissive, which is the safe direction.
    """
    values = [node.get(key) for node in nodes]
    if all(value is None for value in values):
        return None

    first = json.dumps(values[0], sort_keys=True)
    if all(json.dumps(value, sort_keys=True) == first for value in values):
        return values[0]

    conflicts.append(Conflict(path, key, "dropped: documents disagree"))
    return None


def _merge_properties(nodes: list[Schema], path: str, conflicts: list[Conflict]) -> Schema:
    """Combine the fields of several objects. Each field is merged from the documents that have it."""
    names: list[str] = []  # every field name, in the order it first appears
    for node in nodes:
        for name in node["properties"]:
            if name not in names:
                names.append(name)

    merged: Schema = {}
    for name in names:
        same_field = [node["properties"][name] for node in nodes if name in node["properties"]]
        merged[name] = _merge_nodes(same_field, f"{path}.{name}", conflicts)
    return merged


def _merge_required(
    nodes: list[Schema], properties: Schema, path: str, conflicts: list[Conflict]
) -> list[str]:
    """A field stays required only if every document requires it (and actually defines it)."""
    required_sets = [set(node.get("required", [])) for node in nodes]
    required_everywhere = set.intersection(*required_sets)  # required by every document
    required_somewhere = set.union(*required_sets)  # required by at least one document

    if required_everywhere != required_somewhere:
        detail = f"kept {len(required_everywhere)} of {len(required_somewhere)} required fields"
        conflicts.append(Conflict(path, "required", detail))

    undefined = required_everywhere - set(properties)
    if undefined:
        detail = f"dropped {sorted(undefined)}: listed as required but never defined"
        conflicts.append(Conflict(path, "required", detail))

    # Keep the same order as the fields themselves.
    return [name for name in properties if name in required_everywhere]


def _most_common(texts: list[str]) -> str:
    """The most frequent text. Ties go to the alphabetically first one, so the result never
    depends on the order of the input."""
    counts = Counter(texts)
    # Sort by count (highest first), then alphabetically, and take the first entry.
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ranked[0][0]


# --- Allowing null -------------------------------------------------------------------------


def _make_fields_nullable(node: Schema) -> None:
    """Make every field below ``node`` accept null. Changes ``node`` in place."""
    for field in node.get("properties", {}).values():
        _add_null(field)
        _make_fields_nullable(field)

    if "items" in node:
        # The list entry itself stays non-nullable, but the fields inside each entry don't.
        _make_fields_nullable(node["items"])


def _add_null(field: Schema) -> None:
    """Let one field accept null, unless the benchmark marks it as never-null."""
    if field.get("not") == NOT_NULL:
        return

    if "type" in field:
        types = sorted(_types_of(field))
        if "null" not in types:
            field["type"] = types + ["null"]

    # An enum lists every allowed value, so null has to be added there too.
    if "enum" in field and None not in field["enum"]:
        field["enum"].append(None)
