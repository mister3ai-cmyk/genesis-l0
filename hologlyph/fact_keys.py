"""
fact_keys.py - key_id for supersession: an explicit source field, or a narrow "object + attribute" rule.
2026-11-21 · 3 functions + 2 constants.

STATUS: EXPERIMENTAL, OFF BY DEFAULT, FROZEN at stage 3 (commit 568a8d2; rule set R1-R3).
  - Nothing in the ingest calls fact_attribute; key_id is set only from an explicit source field.
  - Do NOT tune these rules on dev LongMemEval any more: R4 was already removed after looking at
    dev false pairs, so dev numbers for this rule are optimistic. A change needs data the rule
    has not seen (test split or keys from a real source).
NOT all regex entities: a key must name ONE updatable fact of the
speaker, or two different facts would hide each other.

Rule (first-person user statements only; first match wins, in this order; every rule needs a value):
  R1 record / amount  "my personal best ... 25:50", "I set a new record of 3:10"   -> "personal best", "record"
  R2 move / job       "I moved to ...", "I now work at ..."                         -> "residence", "employer"
  R3 count of a thing "I've completed 50 episodes of ...", "I lead a team of 4 engineers"
                      -> "episodes of science series", "engineers"
                      (verb of possession / progress required; generic plurals like minutes,
                      tips, ideas rejected; fillers like "more", "new" dropped)
  ("my new phone / apartment ..." is NOT a key: it names an object but no value of a fact;
   on dev LongMemEval it produced most of the false supersessions, so that rule was removed.)
key_id = crc32(namespace | attribute), never 0. The namespace is the owner of the facts
(user / source id) so keys never collide across owners.
"""
import re
import zlib
from typing import Optional

EXPERIMENTAL = True
FROZEN_AT = "568a8d2"           # stage 3; rules R1-R3 unchanged since

_NUM = (r"(?:\d[\d,.:]*|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|"
        r"fifty|a dozen)")
_R1 = re.compile(r"(?:\bmy (?:new |current |previous )?|\bi(?: [a-z]+){0,2} (?:set|hit|reached|beat|broke) "
                 r"(?:a |my )?(?:new )?)"
                 r"(personal best(?: time)?|record|goal|target|budget|salary|rent|weight|savings|score|commute|mortgage)"
                 r"\b[^.!?\n]{0,60}?(?:\$\s?)?\d")
_R2 = [(re.compile(r"\bi(?:'ve| have)? (?:just |recently |finally |now )?(?:moved|relocated) to\b"), "residence"),
       (re.compile(r"\bi (?:now |currently |just )?(?:started )?(?:work|working) (?:at|for)\b"), "employer")]
_VERB = (r"(?:have|has|had|own|owned|lead|led|manage|complete|completed|finish|finished|attend|attended|write|"
         r"written|wrote|read|watched|collected|saved|need|needed|ran|run|visited|bought|adopted|taken|took|made|"
         r"baked|planted|tried)")
_R3 = re.compile(r"\bi(?:'ve|'m| have| am)?(?: [a-z]+){0,2} " + _VERB +
                 r"(?: [a-z]+){0,2}? (?:a total of |a team of |a group of |about |around |over |only |just |exactly )?"
                 + _NUM + r" ((?:[a-z]+ ){0,2}[a-z]{3,}s)\b(?: of (?:the |my )?([a-z]+(?: [a-z]+){0,2}))?")
_GENERIC = set("minutes hours days weeks months years seconds times tips ways steps ideas things people examples "
               "options questions suggestions words points reasons pages lines items parts kinds types sorts lots bits "
               "hundreds thousands dollars euros pounds miles kilometers others yes thanks".split())
_FILLER = set("more new other additional different extra total of the my".split())


def fact_attribute(text: str, role: str = "user") -> Optional[str]:
    """The single updatable attribute a first-person statement talks about, or None."""
    if role != "user":
        return None
    t = text.lower()
    m = _R1.search(t)
    if m:
        return m.group(1).replace(" time", "")
    for rx, name in _R2:
        if rx.search(t):
            return name
    for m in _R3.finditer(t):
        words = [w for w in m.group(1).split() if w not in _FILLER]
        if not words or words[-1] in _GENERIC:
            continue
        of = m.group(2)
        return " ".join(words) + (" of " + of if of else "")
    return None


def key_id(namespace: str, attribute: Optional[str]) -> int:
    """crc32(namespace | attribute) as a non-zero u32; 0 when there is no attribute."""
    if not attribute:
        return 0
    k = zlib.crc32(("%s|%s" % (namespace, attribute)).encode("utf-8")) & 0xFFFFFFFF
    return k or 1


def key_for(text: str, namespace: str, role: str = "user", explicit: Optional[str] = None,
            use_rule: bool = False) -> int:
    """Explicit source field (e.g. a CRM field name) -> key; the experimental rule only with use_rule=True."""
    if explicit:
        return key_id(namespace, explicit)
    return key_id(namespace, fact_attribute(text, role)) if use_rule else 0
