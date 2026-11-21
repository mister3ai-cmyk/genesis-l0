"""
test_fact_keys.py - the "object + attribute" key rule: hits, rejections, namespaces.
2026-11-21 · 3 tests (pure Python).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fact_keys as fk  # noqa: E402


def test_rule_hits_one_attribute_with_a_value():
    cases = {
        "I recently set a personal best time in a charity 5K run with a time of 27:12.": "personal best",
        "I'm hoping to beat my personal best time of 25:50 this time around.": "personal best",
        "I have completed 50 episodes of the Science series": "episodes of science series",
        "I lead a team of 4 engineers in my new role": "engineers",
        "I did attend three sessions of the bereavement support group": "sessions of bereavement support group",
        "I have written five more short stories": "short stories",
        "I just moved to Denver last month": "residence",
    }
    for text, attr in cases.items():
        assert fk.fact_attribute(text) == attr, text


def test_rule_rejects_generic_valueless_and_assistant():
    for text in ("It takes me 20 minutes", "Here are 5 tips for you", "I love my new phone",
                 "I'm looking for yoga classes near my new apartment", "thanks!"):
        assert fk.fact_attribute(text) is None, text
    assert fk.fact_attribute("I have completed 50 episodes", role="assistant") is None


def test_key_ids_namespaced_and_nonzero():
    a = fk.key_for("I lead a team of 4 engineers", "user-1")
    b = fk.key_for("Now I lead 5 engineers", "user-1")
    c = fk.key_for("I lead a team of 4 engineers", "user-2")
    assert a == b != 0 and a != c
    assert fk.key_for("hello", "user-1") == 0
    assert fk.key_for("anything", "user-1", explicit="crm.phone") == fk.key_id("user-1", "crm.phone") != 0
