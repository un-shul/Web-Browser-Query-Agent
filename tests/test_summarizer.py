"""Tests for text cleaning, query-aware selection, and chunking.

None of these load the summarisation model. The tokeniser-dependent path in
split_text has a word-count fallback, so chunking is exercised through that.
"""

import pytest

from queryagent.summarize import local as S


# --- clean_text --------------------------------------------------------------


@pytest.mark.parametrize(
    "term",
    ["light-dependent", "3-phosphoglyceric", "energy-carrying", "carbon-fixing"],
)
def test_hyphenated_terms_survive(term):
    """Regression: the allowed-character class excluded '-', so real output
    contained 'lightdependent' and '3phosphoglyceric'."""
    assert term in S.clean_text(f"The {term} reaction is central to the process.")


def test_percentages_and_parentheses_survive():
    out = S.clean_text("Roughly 30% (thirty percent) of the energy is lost.")
    assert "30%" in out
    assert "(" in out


def test_proper_nouns_survive():
    """Regression: any sentence with >40% capitalised words was deleted, which
    removed most content about people and places."""
    text = "Alan Turing was born in London and studied at King's College Cambridge."
    out = S.clean_text(text)
    assert "Alan Turing" in out
    assert "London" in out


def test_boilerplate_is_removed():
    out = S.clean_text(
        "Photosynthesis converts light. We use cookies to improve your "
        "experience. Click here to subscribe. All rights reserved."
    )
    assert "Photosynthesis converts light" in out
    for junk in ("cookies", "Click here", "rights reserved"):
        assert junk.lower() not in out.lower()


def test_navigation_runs_are_removed():
    out = S.clean_text(
        "Home News Sport Business Travel Culture Weather Photosynthesis is a "
        "process that converts light into chemical energy."
    )
    assert "Photosynthesis" in out


def test_clean_text_is_idempotent():
    once = S.clean_text("The light-dependent reaction yields 30% more ATP.")
    assert S.clean_text(once) == once


# --- query-aware selection ---------------------------------------------------

DOC = (
    "Photosynthesis converts light energy into chemical energy. "
    "Chlorophyll absorbs red and blue wavelengths. "
    "ATP then drives the Calvin cycle to build sugar. "
    "Deforestation in the Amazon is driven by cattle ranching. "
    "The share price of the timber company rose last quarter."
)


def test_irrelevant_sentences_rank_below_relevant_ones():
    selected = S.select_relevant(DOC, "what is photosynthesis", keep_chars=130)
    assert "share price" not in selected
    assert "Deforestation" not in selected


def test_neighbour_credit_retains_explanatory_sentences():
    """The point of neighbour credit.

    "Chlorophyll absorbs red and blue wavelengths" contains no query term for
    "what is photosynthesis", so pure term overlap scores it zero and ranks it
    alongside unrelated filler. It is the actual explanation, so it must be
    kept.
    """
    selected = S.select_relevant(DOC, "what is photosynthesis", keep_chars=130)
    assert "Chlorophyll" in selected


def test_selection_preserves_document_order():
    selected = S.select_relevant(DOC, "chlorophyll photosynthesis", keep_chars=200)
    assert selected.index("Photosynthesis converts") < selected.index("Chlorophyll absorbs")


def test_no_query_returns_a_prefix():
    assert S.select_relevant(DOC, None, keep_chars=60) == DOC[:60]


def test_query_of_only_stopwords_is_treated_as_no_query():
    assert S._terms("what is the") == set()


def test_short_query_terms_are_ignored():
    # Two-character tokens are noise, not topic signal.
    assert "of" not in S._terms("cost of ai")
    assert "ai" not in S._terms("cost of ai")


# --- chunking ----------------------------------------------------------------


def test_chunks_respect_the_token_budget():
    """Regression: chunking counted words against a token limit.

    380 words of technical prose tokenised to 1152 tokens against distilbart's
    1024-token encoder, and the pipeline raised "index 1026 is out of bounds".
    That was caught and logged as a skipped chunk, silently dropping the
    content from the summary.
    """
    text = " ".join(f"Sentence number {i} about photosynthesis." for i in range(400))
    chunks = list(S.split_text(text, max_tokens=200))
    assert len(chunks) > 1
    for chunk in chunks:
        # The fallback estimator counts 2 tokens/word, so this is the bound it
        # guarantees.
        assert len(chunk.split()) * 2 <= 200 * 2


def test_chunking_loses_no_sentences():
    text = " ".join(f"Fact {i} is true." for i in range(120))
    rejoined = " ".join(S.split_text(text, max_tokens=120))
    for i in (0, 60, 119):
        assert f"Fact {i} is true." in rejoined


def test_oversized_single_sentence_is_split_not_dropped():
    giant = "word " * 3000
    chunks = list(S.split_text(giant.strip(), max_tokens=100))
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)


def test_empty_text_yields_no_usable_chunks():
    assert [c for c in S.split_text("", max_tokens=100) if c.strip()] == []


# --- dedupe and trimming -----------------------------------------------------


def test_dedupe_collapses_repeated_openers():
    """Each chunk is summarised independently, so chunks covering the same
    material repeat the same opening sentence. Real output repeated
    "Photosynthesis is the process by which..." three times."""
    out = S._dedupe([
        "Photosynthesis is the process by which plants make sugar.",
        "Photosynthesis is the process by which plants make sugar.",
        "Chlorophyll gives plants their green colour.",
    ])
    assert len(out) == 2


def test_dedupe_keeps_genuinely_different_sentences():
    out = S._dedupe(["Plants need light.", "Plants need water.", "Plants need carbon dioxide."])
    assert len(out) == 3


def test_dedupe_preserves_order():
    out = S._dedupe(["First point here.", "Second point here.", "First point here."])
    assert out == ["First point here.", "Second point here."]


def test_trim_stops_on_a_sentence_boundary():
    text = "One sentence here. Two sentence here. Three sentence here."
    out = S.trim_to_sentence_boundary(text, max_chars=40)
    assert out.endswith(".")
    assert len(out) <= 45


def test_trim_leaves_short_text_alone():
    assert S.trim_to_sentence_boundary("Short.", max_chars=100) == "Short."


def test_short_input_is_reported_not_summarised():
    # Must not load the model for input that cannot be summarised.
    assert S.summarize_text("Too short.", "query") == "Content too short to summarize."


def test_fix_punctuation_capitalises_after_periods():
    assert S.fix_punctuation("one thing. two thing.") == "One thing. Two thing."


# --- markdown and unicode ----------------------------------------------------
# Provider-sourced pages arrive as markdown, so these paths only trigger when
# the scrape fallback chain reaches Tavily's raw_content.


def test_unicode_apostrophe_is_preserved_as_ascii():
    """Regression: U+2019 fell outside the allowed-character class and became a
    space, so "Earth's atmosphere" rendered as "Earth s atmosphere"."""
    assert "Earth's" in S.clean_text("Earth’s atmosphere")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("“quoted”", '"quoted"'),
        ("a — b", "a - b"),
        ("a – b", "a - b"),
        ("wait… ok", "wait... ok"),
    ],
)
def test_typographic_punctuation_is_normalised(raw, expected):
    assert expected in S.clean_text(f"Sentence with {raw} inside.")


def test_non_breaking_space_becomes_a_space():
    assert "10 kg" in S.clean_text("The mass is 10 kg total.")


def test_markdown_link_keeps_text_and_drops_target():
    """Regression: allowing ':' and '/' for prose let URLs through, producing
    "English Learners in STEM (https: //ssec. Si. Edu/...)" in a summary."""
    out = S.clean_text("See [English Learners in STEM](https://ssec.si.edu/x) for more.")
    assert "English Learners in STEM" in out
    assert "ssec.si.edu" not in out


def test_bare_url_is_removed():
    out = S.clean_text("Read https://ssec.si.edu/stemvisions-blog/what-photosynthesis today.")
    assert "http" not in out
    assert "Read" in out and "today" in out


def test_markdown_image_is_removed_entirely():
    out = S.clean_text("![diagram](https://x.com/a.png) Photosynthesis converts light.")
    assert "diagram" not in out
    assert "Photosynthesis converts light" in out


def test_markdown_heading_and_bullet_markers_are_stripped():
    out = S.clean_text("## What is Photosynthesis\n- Plants use light.")
    assert "#" not in out and not out.lstrip().startswith("-")
    assert "Plants use light" in out


def test_legitimate_punctuation_still_survives_url_stripping():
    out = S.clean_text("Roughly 30% of light-dependent energy (NADPH) is lost.")
    for token in ("30%", "light-dependent", "(NADPH)"):
        assert token in out


def test_period_restored_after_closing_paren():
    """The model sometimes ends a sentence on ')' and drops the period."""
    out = S.fix_punctuation("chemical energy (sugar) The oxygen we breathe is free.")
    assert "(sugar). The oxygen" in out


def test_paren_followed_by_lowercase_is_left_alone():
    out = S.fix_punctuation("oxygen (O 2 ) and energy stored in glucose.")
    assert "). and" not in out
