"""
utils/helpers.py — Small reusable utilities shared across modules.
"""

import re
import unicodedata


def normalize_entity_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", name)
    name = name.lower().strip()
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name


def relationship_type_to_verb(rel_type: str) -> str:
    return rel_type.lower().replace("_", " ")


def serialize_triple(source: str, relation_type: str, target: str) -> str:
    verb = relationship_type_to_verb(relation_type)
    return f"{source} {verb} {target}."


def chunk_id(page_slug: str, index: int) -> str:
    return f"{page_slug}_{index:03d}"