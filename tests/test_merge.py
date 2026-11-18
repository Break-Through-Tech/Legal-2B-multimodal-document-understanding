"""Unit tests for schemas/merge.py.

Run from the repository root with:  pytest
"""

import copy

import pytest
from jsonschema import Draft202012Validator

from schemas.merge import allow_null, merge_schemas

# Small building blocks used throughout the tests.
STRING = {"type": "string"}
NUMBER = {"type": "number"}


def obj(properties, required=()):
    """Build an object schema, e.g. obj({"name": STRING}, required=["name"])."""
    return {"type": "object", "properties": properties, "required": list(required)}


def accepts(schema, instance):
    return Draft202012Validator(schema).is_valid(instance)


# --- merge_schemas -------------------------------------------------------------------------


def test_identical_schemas_merge_to_the_same_schema_but_closed():
    schema = obj({"name": STRING, "amount": NUMBER}, required=["name"])
    merged, conflicts = merge_schemas([schema, copy.deepcopy(schema)])
    assert merged == {**schema, "additionalProperties": False}
    assert conflicts == []


def test_fields_are_combined_and_required_is_shared_only():
    a = obj({"name": STRING, "city": STRING}, required=["name", "city"])
    b = obj({"name": STRING, "zip": STRING}, required=["name"])
    merged, conflicts = merge_schemas([a, b])
    assert list(merged["properties"]) == ["name", "city", "zip"]  # order of first appearance
    assert merged["required"] == ["name"]  # "city" isn't required in b
    assert [c.keyword for c in conflicts] == ["required"]


def test_objects_are_closed_to_fields_seen_in_training():
    merged, _ = merge_schemas([obj({"name": STRING})])
    assert accepts(merged, {"name": "Acme"})
    assert not accepts(merged, {"name": "Acme", "invented": "x"})


def test_integer_folds_into_number_but_other_types_widen():
    merged, conflicts = merge_schemas([{"type": "integer"}, NUMBER])
    assert merged == {"type": "number"}
    assert conflicts == []

    merged, conflicts = merge_schemas([STRING, NUMBER])
    assert merged["type"] == ["number", "string"]
    assert conflicts[0].keyword == "type"


def test_enums_are_combined_or_dropped_when_any_input_allows_anything():
    a = {"type": "string", "enum": ["credit", "debit"]}
    b = {"type": "string", "enum": ["deposit", "credit"]}
    merged, _ = merge_schemas([a, b])
    assert merged["enum"] == ["credit", "debit", "deposit"]

    merged, conflicts = merge_schemas([a, STRING])
    assert "enum" not in merged
    assert conflicts[0].keyword == "enum"


def test_enums_that_differ_only_in_order_are_not_a_conflict():
    a = {"type": "string", "enum": ["A", "B"]}
    b = {"type": "string", "enum": ["B", "A"]}
    _, conflicts = merge_schemas([a, b])
    assert conflicts == []


def test_enum_values_true_and_1_stay_distinct():
    merged, _ = merge_schemas([{"type": ["boolean", "number"], "enum": [True, 1]}])
    assert merged["enum"] == [True, 1]


def test_constraints_survive_only_when_every_input_agrees():
    dated = {"type": "string", "format": "date"}
    merged, _ = merge_schemas([dated, copy.deepcopy(dated)])
    assert merged["format"] == "date"

    merged, conflicts = merge_schemas([{"type": "string", "pattern": "^Q[1-4]$"}, STRING])
    assert "pattern" not in merged
    assert conflicts[0].keyword == "pattern"


def test_array_entries_are_merged_too():
    a = {"type": "array", "items": obj({"x": NUMBER}, required=["x"])}
    b = {"type": "array", "items": obj({"y": NUMBER})}
    merged, _ = merge_schemas([a, b])
    assert list(merged["items"]["properties"]) == ["x", "y"]
    assert "required" not in merged["items"]


def test_required_field_without_a_definition_is_dropped_and_reported():
    schema = {"type": "object", "properties": {"a": STRING}, "required": ["a", "ghost"]}
    merged, conflicts = merge_schemas([schema])
    assert merged["required"] == ["a"]
    assert any("ghost" in c.detail for c in conflicts)


def test_required_on_a_list_is_ignored_and_reported():
    # PROXY_VOTING's schemas do this; "required" has no effect on a list.
    schema = {"type": "array", "items": obj({"name": STRING}), "required": ["name"]}
    merged, conflicts = merge_schemas([schema])
    assert "required" not in merged
    assert conflicts[0].detail == "ignored: 'required' only applies to objects"


def test_merged_schema_accepts_what_each_input_accepts_using_its_declared_fields():
    a = obj({"kind": {"type": "string", "enum": ["credit"]}, "amount": {"type": "integer"}},
            required=["kind", "amount"])
    b = obj({"kind": {"type": "string", "enum": ["deposit"]}, "note": STRING}, required=["kind"])
    merged, _ = merge_schemas([a, b])
    assert accepts(merged, {"kind": "credit", "amount": 5})  # valid under a
    assert accepts(merged, {"kind": "deposit", "note": "x"})  # valid under b
    assert not accepts(merged, {"kind": "refund"})  # no input allowed "refund"


def test_merge_does_not_change_its_inputs_or_share_pieces_with_them():
    a = obj({"status": {"type": "string", "enum": ["A"]}})
    original = copy.deepcopy(a)
    merged, _ = merge_schemas([a])
    merged["properties"]["status"]["enum"].append("B")
    assert a == original


@pytest.mark.parametrize(
    "bad_schema",
    [
        {"type": "enum", "enum": ["A"]},  # "enum" is not a JSON Schema type (seen in BANK_CHECK)
        {"type": "string", "emum": ["A"]},  # misspelled keyword (seen in PATIENT_INTAKE)
        {"description": "no type"},  # missing type
        {"type": "object"},  # object without properties
        {"type": "array"},  # array without items
        {"type": "object", "properties": {}, "additionalProperties": STRING},  # schema-valued
        {"type": "array", "items": True},  # boolean sub-schema
    ],
)
def test_unsupported_schema_content_fails_loudly(bad_schema):
    # ValueError for bad content, TypeError when a value has the wrong Python type.
    with pytest.raises((ValueError, TypeError)):
        merge_schemas([bad_schema])


def test_empty_input_is_rejected():
    with pytest.raises(ValueError):
        merge_schemas([])


# --- allow_null ----------------------------------------------------------------------------


def test_allow_null_makes_fields_nullable_except_root_list_entries_and_never_null_fields():
    schema = obj({
        "name": STRING,
        "status": {"type": "string", "enum": ["A", "B"]},
        "blind": {"type": "boolean", "not": {"type": "null"}},
        "rows": {"type": "array", "items": obj({"value": NUMBER})},
    })
    result = allow_null(schema)
    fields = result["properties"]

    assert result["type"] == "object"  # root unchanged
    assert fields["name"]["type"] == ["string", "null"]
    assert fields["status"]["enum"] == ["A", "B", None]
    assert fields["blind"]["type"] == "boolean"  # marked never-null
    assert fields["rows"]["items"]["type"] == "object"  # list entries unchanged
    assert fields["rows"]["items"]["properties"]["value"]["type"] == ["number", "null"]


def test_allow_null_returns_an_independent_copy():
    schema = obj({"blind": {"type": "boolean", "not": {"type": "null"}}})
    original = copy.deepcopy(schema)
    result = allow_null(schema)
    result["properties"]["blind"]["not"]["type"] = "string"
    assert schema == original
