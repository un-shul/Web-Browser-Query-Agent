"""The deterministic checks for what embeddings miss.

Measured cosines on all-MiniLM-L6-v2 appear in the docstrings; these tests
pin the behaviour those measurements justified.
"""

import pytest

from queryagent.llm import mismatch as M


@pytest.mark.parametrize("a,b", [
    ("is coffee good for your health", "is coffee bad for your health"),
    ("advantages of remote work", "disadvantages of remote work"),
    ("how to install docker", "how to uninstall docker"),
    ("best programming language", "worst programming language"),
    ("how to enable ipv6", "how to disable ipv6"),
    ("benefits of fasting", "risks of fasting"),
    ("why you should buy gold", "why you should sell gold"),
    ("is it safe to eat raw eggs", "is it dangerous to eat raw eggs"),
])
def test_polarity_conflicts_are_detected(a, b):
    assert M.polarity_conflict(a, b)


@pytest.mark.parametrize("a,b", [
    ("what is photosynthesis", "explain photosynthesis"),
    ("how does a car engine work", "explain how car engines work"),
    ("capital of france", "which city is france's capital"),
])
def test_equivalent_paraphrases_are_not_conflicts(a, b):
    assert M.find_conflict(a, b) is None


def test_negation_on_one_side_only():
    assert M.polarity_conflict("is coffee healthy", "is coffee not healthy")


def test_negation_on_both_sides_is_not_a_conflict():
    """"not X" vs "not Y" is a real pair of questions, not a polarity flip."""
    assert M.polarity_conflict("foods that are not safe", "drinks that are not safe") is None


@pytest.mark.parametrize("a,b", [
    ("2024 election results", "2025 election results"),
    ("best laptops under 50000", "best laptops under 100000"),
    ("symptoms of type 1 diabetes", "symptoms of type 2 diabetes"),
    ("top 5 movies", "top 10 movies"),
])
def test_quantity_conflicts_are_detected(a, b):
    assert M.quantity_conflict(a, b)


def test_identical_numbers_are_not_a_conflict():
    assert M.quantity_conflict("top 10 movies of 2024", "best 10 films of 2024") is None


def test_no_numbers_anywhere_is_not_a_conflict():
    assert M.quantity_conflict("what is photosynthesis", "explain photosynthesis") is None


@pytest.mark.parametrize("a,b", [
    ("flights from delhi to mumbai", "flights from mumbai to delhi"),
    ("convert usd to inr", "convert inr to usd"),
    ("trains from pune to nagpur", "trains from nagpur to pune"),
])
def test_direction_reversal_is_detected(a, b):
    """These sit at ~0.997 cosine, so retrieval gives no signal at all."""
    assert M.direction_conflict(a, b)


def test_same_direction_is_not_a_conflict():
    assert M.direction_conflict("flights from delhi to mumbai",
                                "cheap flights from delhi to mumbai") is None


def test_direction_check_needs_a_directional_word():
    assert M.direction_conflict("mumbai delhi flights", "delhi mumbai flights") is None


# --- hard vs verifier-decided ------------------------------------------------


@pytest.mark.parametrize("a,b", [
    ("is coffee good for your health", "is coffee bad for your health"),
    ("flights from delhi to mumbai", "flights from mumbai to delhi"),
])
def test_polarity_and_direction_are_hard_conflicts(a, b):
    """Decided locally, because the verifier rationalises both -- see
    mismatch.is_hard_conflict for the measurements."""
    assert M.is_hard_conflict(a, b)


@pytest.mark.parametrize("a,b", [
    ("2024 election results", "2025 election results"),
    ("best laptops under 50000", "best laptops under 100000"),
])
def test_quantity_is_left_to_the_verifier(a, b):
    """The verifier judges these correctly, so it keeps the decision."""
    assert M.is_hard_conflict(a, b) is None
    assert M.find_conflict(a, b)


def test_safe_to_auto_accept_blocks_a_conflicting_pair():
    safe, reason = M.safe_to_auto_accept("is coffee good for you", "is coffee bad for you")
    assert not safe and reason


def test_safe_to_auto_accept_allows_a_clean_pair():
    safe, reason = M.safe_to_auto_accept("what is photosynthesis", "explain photosynthesis")
    assert safe and reason == ""


@pytest.mark.parametrize("a,b", [("", ""), ("x", ""), ("", "y"), ("   ", "  ")])
def test_empty_input_is_handled(a, b):
    assert M.find_conflict(a, b) is None
