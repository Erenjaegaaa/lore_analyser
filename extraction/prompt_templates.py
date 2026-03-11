"""
extraction/prompt_templates.py — Gemini extraction prompt builder.

Builds the prompt sent to Gemini for entity + relationship extraction.
The prompt ends with `Output:` so Gemini completes directly into JSON.
All literal braces are doubled so the f-string doesn't misinterpret them.
"""


def build_extraction_prompt(chunk_text: str) -> str:
    """
    Return a Gemini prompt that extracts entities and relationships from
    *chunk_text* and returns only valid JSON — no markdown, no preamble.

    Args:
        chunk_text: Plain-text lore passage to analyse.

    Returns:
        Fully-formed prompt string ending with ``Output:``
    """
    return f"""You are a knowledge-graph extraction engine for Tolkien / Middle-earth lore.

Given a passage of lore text, extract:
1. Named **entities** (characters, locations, events, artifacts, factions).
2. **Relations** between those entities using ONLY the 18 predicates listed below.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VALID ENTITY TYPES (use exactly these strings):
  Character | Location | Event | Artifact | Faction

VALID PREDICATES (18 total — use ONLY these, spelled exactly):

  # Kinship
  CHILD_OF         — lineage: child → parent (e.g. Elrond CHILD_OF Eärendil)
  SIBLING_OF       — shared parentage (e.g. Elrond SIBLING_OF Elros)
  SPOUSE_OF        — married / bonded pair
  HEIR_OF          — designated successor (e.g. Aragorn HEIR_OF Isildur)

  # Alliance & Enmity
  ALLY_OF          — allied characters or factions
  ENEMY_OF         — opposed characters or factions
  SERVANT_OF       — subject / thrall (e.g. Grima SERVANT_OF Saruman)

  # Faction & Politics
  MEMBER_OF        — character belongs to a faction
  RULES_OVER       — character holds dominion over a location or faction

  # Craftsmanship & Artifacts
  CREATED          — general creation (wrote a book, built a city)
  FORGED_BY        — direction is Artifact → Character (e.g. One Ring FORGED_BY Sauron)
  WIELDED          — character carried / used an artifact

  # Geography
  BORN_IN          — character's place of origin
  PART_OF          — direction is smaller → larger (e.g. Shire PART_OF Eriador)
  LOCATED_IN       — artifact physically resides at a location

  # Events
  OCCURRED_AT      — event took place at a location
  PARTICIPATED_IN  — character took part in an event (or event involved a character)
  RESULTED_IN      — event caused another event or produced an artifact
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OUTPUT FORMAT — return ONLY this JSON, no markdown fences, no explanation:
{{
  "entities": [
    {{"name": "<string>", "type": "<EntityType>", "aliases": ["<alt name>", ...]}}
  ],
  "relations": [
    {{"subject": "<name>", "predicate": "<PREDICATE>", "object": "<name>"}}
  ]
}}

Rules:
- If a field has no values, use an empty list [].
- Only extract entities and relations that are explicitly supported by the text.
- Do not invent information not present in the passage.
- Prefer the most specific predicate that fits.
- For FORGED_BY the subject must be the Artifact, not the maker.
- For PART_OF the subject must be the contained location, object the containing one.
- Aliases should only include names that appear in this passage.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Example 1
Input: "Aragorn, also known as Strider and Elessar, is the heir of Isildur and a
member of the Dúnedain. He was born in Rivendell and was a close ally of Gandalf.
He wielded Andúril, the sword reforged from the shards of Narsil."

Output:
{{
  "entities": [
    {{"name": "Aragorn", "type": "Character", "aliases": ["Strider", "Elessar"]}},
    {{"name": "Isildur", "type": "Character", "aliases": []}},
    {{"name": "Dúnedain", "type": "Faction", "aliases": []}},
    {{"name": "Rivendell", "type": "Location", "aliases": []}},
    {{"name": "Gandalf", "type": "Character", "aliases": []}},
    {{"name": "Andúril", "type": "Artifact", "aliases": ["Narsil"]}}
  ],
  "relations": [
    {{"subject": "Aragorn", "predicate": "HEIR_OF", "object": "Isildur"}},
    {{"subject": "Aragorn", "predicate": "MEMBER_OF", "object": "Dúnedain"}},
    {{"subject": "Aragorn", "predicate": "BORN_IN", "object": "Rivendell"}},
    {{"subject": "Aragorn", "predicate": "ALLY_OF", "object": "Gandalf"}},
    {{"subject": "Aragorn", "predicate": "WIELDED", "object": "Andúril"}}
  ]
}}

Example 2
Input: "The One Ring was forged by Sauron in the fires of Mount Doom, deep within
the land of Mordor. Mount Doom itself lies in the Plateau of Gorgoroth, which is
a region of Mordor. During the War of the Last Alliance, the Ring was cut from
Sauron's hand by Isildur."

Output:
{{
  "entities": [
    {{"name": "One Ring", "type": "Artifact", "aliases": []}},
    {{"name": "Sauron", "type": "Character", "aliases": []}},
    {{"name": "Mount Doom", "type": "Location", "aliases": []}},
    {{"name": "Mordor", "type": "Location", "aliases": []}},
    {{"name": "Plateau of Gorgoroth", "type": "Location", "aliases": ["Gorgoroth"]}},
    {{"name": "War of the Last Alliance", "type": "Event", "aliases": []}},
    {{"name": "Isildur", "type": "Character", "aliases": []}}
  ],
  "relations": [
    {{"subject": "One Ring", "predicate": "FORGED_BY", "object": "Sauron"}},
    {{"subject": "Mount Doom", "predicate": "PART_OF", "object": "Mordor"}},
    {{"subject": "Plateau of Gorgoroth", "predicate": "PART_OF", "object": "Mordor"}},
    {{"subject": "Isildur", "predicate": "PARTICIPATED_IN", "object": "War of the Last Alliance"}},
    {{"subject": "Sauron", "predicate": "PARTICIPATED_IN", "object": "War of the Last Alliance"}}
  ]
}}

Example 3
Input: "The Shire is a peaceful region of Eriador inhabited by Hobbits. Frodo Baggins,
son of Drogo Baggins, lived in the Shire at Bag End. The Fellowship of the Ring was
formed in Rivendell to counter the threat of Sauron. Saruman, once an ally of Gandalf,
became an enemy of the Free Peoples after his corruption."

Output:
{{
  "entities": [
    {{"name": "Shire", "type": "Location", "aliases": []}},
    {{"name": "Eriador", "type": "Location", "aliases": []}},
    {{"name": "Frodo Baggins", "type": "Character", "aliases": ["Frodo"]}},
    {{"name": "Drogo Baggins", "type": "Character", "aliases": []}},
    {{"name": "Fellowship of the Ring", "type": "Faction", "aliases": ["the Fellowship"]}},
    {{"name": "Rivendell", "type": "Location", "aliases": []}},
    {{"name": "Sauron", "type": "Character", "aliases": []}},
    {{"name": "Saruman", "type": "Character", "aliases": []}},
    {{"name": "Gandalf", "type": "Character", "aliases": []}}
  ],
  "relations": [
    {{"subject": "Shire", "predicate": "PART_OF", "object": "Eriador"}},
    {{"subject": "Frodo Baggins", "predicate": "CHILD_OF", "object": "Drogo Baggins"}},
    {{"subject": "Saruman", "predicate": "ENEMY_OF", "object": "Gandalf"}}
  ]
}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NOW EXTRACT FROM THIS PASSAGE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input: "{chunk_text}"

Output:"""