"""Deterministic screen of explanation text against the evidence the model was shown.

WHAT THIS IS. The explain prompt tells the model to "cite only titles, ratings, genres, tags, and
answers shown above". This module checks that, mechanically: every title, rating, genre, quoted
tag, decade and reference to a clarification in an explanation must trace back to the profile text,
the clarification answers, or the catalog line of a movie that was displayed.

WHAT THIS IS NOT. It cannot decide whether an explanation is true or sensible. A text that passes
(``passed_screen``) merely contains no *detectable* claim beyond the displayed evidence. Prose such
as "you will love the pacing" is not checkable here and is not flagged. Semantic truth, whether a
simulated answer used only pre-cutoff history, and whether the model's decisions were autonomous
need the human audit in ``docs/phase2_grounding_audit.md`` (Member 2). Statuses use that audit's
vocabulary so the two line up, but this module never says "supported": that word is the reviewer's.

The rule set is a proposal (decision A15). It was written to accept Member 1's template
explanations and to flag the fabrications the audit rubric names.

KNOWN LIMITS (deliberate, documented, tested where noted):
* Prose claims that name no title, rating, genre, tag, decade or answer are not checked.
* Titles are matched exactly against the profile, the simulator's answers and the displayed
  movies. A title in a FREE-TEXT answer is matched leniently: a trailing run of words plus the
  year found in an answer is accepted, so a partial title ("Street (1994)") can slip through.
* A cited title is a name, not a claim: its words ("Mystery" in "Mystery Men (1999)") are left out of the
  genre and tag scans. A title written without its "(YYYY)" is not recognised as a title.
* Only genres, quoted tags, "(YYYY)" titles, "N/5" ratings, "1990s"-style decades, "you said"-style
  references and a short list of unavailable fields (director, cast, plot, ...) are looked for.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from src.data.p2_checks import GENRES  # the 19 MovieLens genre names, shared with Member 1's checker

UNGROUNDED = "ungrounded_explanation"  # reason code when the reject policy refuses an explanation set

# Result statuses, strongest first. Names follow Member 2's audit rubric where one exists.
UNSUPPORTED = "unsupported"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"
NEEDS_REVIEW = "needs_review"
NO_CHECKABLE_CLAIM = "no_checkable_claim"
PASSED_SCREEN = "passed_screen"
_SEVERITY = (UNSUPPORTED, INSUFFICIENT_EVIDENCE, NEEDS_REVIEW)

MAX_TEXT = 2000  # longer text is reported, not silently half-checked

_YEAR_CITE = re.compile(r"\((?:1[89]\d\d|20\d\d)\)")
_RATING = re.compile(r"(?<![\w.])(\d(?:\.\d)?)/5(?!\d)")
_DECADE = re.compile(r"(?<![\w])((?:18|19|20)\d0)s(?![\w])")
_QUOTED = re.compile(r'"([^"\n]{1,100})"')
_ANSWER_REF = re.compile(
    r"\b(?:you (?:said|mentioned|told me|stated|noted|answered)|your (?:answers?|repl(?:y|ies))|stated preference)\b",
    re.IGNORECASE)
_UNAVAILABLE = re.compile(
    r"\b(?:directed by|director|starring|cast|actors?|actress(?:es)?|screenplay|written by|box[- ]office|"
    r"grossed|budget|runtime|critics?|reviews?|imdb|rotten tomatoes|metacritic|award[- ]winning|"
    r"oscar[- ]winning|plot|storyline|soundtrack)\b", re.IGNORECASE)
_GENRE_PATTERNS = {genre: re.compile(rf"(?<![\w-]){re.escape(genre)}(?![\w-])") for genre in GENRES}

# Profile text is a fixed template (src/profiles/text.py): "Highly rated movies: T (4/5; genres: ..;
# tags: ..); T2 (...). Low rated movies: T3 (2/5); T4 (1/5)."
_LIKED_PREFIX, _LOW_MARKER = "Highly rated movies: ", " Low rated movies: "
_LIKED_ITEM = re.compile(r"(?:^|; )(?P<title>[^;]+?) \((?P<rating>\d+(?:\.\d+)?)/5; genres: [^;]*?; tags: [^;]*?\)"
                         r"(?=; |\.(?: |$))")
_LOW_ITEM = re.compile(r"(?:^|; )(?P<title>[^;]+?) \((?P<rating>\d+(?:\.\d+)?)/5\)(?=; |\.\s*$)")
# The simulator's answers (src/data/clarification_sim.py answer_text).
_ANSWER_TITLE = re.compile(r"(?:really enjoyed|didn't enjoy) (?P<title>.+?) \((?P<rating>\d+(?:\.\d+)?)/5\)\.")


@dataclass(frozen=True)
class Evidence:
    """Everything the model saw when it wrote the explanations."""

    profile_text: str
    answers: tuple[str, ...]
    shown: Mapping[int, Mapping]  # movie_id -> catalog doc, for the movies listed in the explain prompt


@dataclass(frozen=True)
class _Known:
    title: str
    side: str  # "user" (profile or answer), "self" (the movie being explained), "other" (another shown movie)
    rating: float | None


def _join(values: Sequence[str]) -> str:
    values = list(values)
    return values[0] if len(values) == 1 else ", ".join(values[:-1]) + " and " + values[-1]


def repair_text(doc: Mapping) -> str:
    """A justification built only from the movie's own displayed catalog fields, so it cannot cite
    anything that was not shown. Used by the ``repair`` policy."""
    genres = list(doc.get("genres") or [])
    # Skip tags that would read as a rating, a year citation or a decade, so the repair cannot trip the screen.
    tags = [tag for tag in (doc.get("tags") or [])
            if '"' not in tag and not (_RATING.search(tag) or _YEAR_CITE.search(tag) or _DECADE.search(tag))][:2]
    if not genres and not tags:
        return "It has no listed genre or tag."
    parts = [f"is listed under {_join(genres)}"] if genres else []
    if tags:
        parts.append("is tagged " + _join([f'"{tag}"' for tag in tags]))
    return "It " + " and ".join(parts) + "."


def _known_titles(movie_id: int, evidence: Evidence) -> list[_Known]:
    known: list[_Known] = []
    profile = evidence.profile_text[:20000]
    body = profile[len(_LIKED_PREFIX):] if profile.startswith(_LIKED_PREFIX) else profile
    low_at = body.find(_LOW_MARKER)
    liked_body, low_body = (body, "") if low_at < 0 else (body[:low_at], body[low_at + len(_LOW_MARKER):])
    for pattern, section in ((_LIKED_ITEM, liked_body), (_LOW_ITEM, low_body)):
        for match in pattern.finditer(section):
            known.append(_Known(match["title"], "user", float(match["rating"])))
    for answer in evidence.answers:
        for match in _ANSWER_TITLE.finditer(answer[:2000]):
            known.append(_Known(match["title"], "user", float(match["rating"])))
    for shown_id, doc in evidence.shown.items():
        known.append(_Known(doc["title"], "self" if shown_id == movie_id else "other", None))
    return sorted(known, key=lambda item: -len(item.title))  # longest title first


def _resolve_cite(text: str, end: int, known: list[_Known], answers_folded: str):
    """Which displayed title does the year-citation ending at ``end`` close? Returns
    ``(resolved, start)``: ``resolved`` is a _Known, the string "answer" (free text matched in a
    clarification answer), or None; ``start`` is where the cited title begins in ``text`` (None when
    unresolved), so the caller can tell the title's own words from the explanation's prose."""
    prefix = text[:end]
    for item in known:
        start = end - len(item.title)
        if start >= 0 and prefix.endswith(item.title) and (start == 0 or not text[start - 1].isalnum()):
            return item, start
    year = text[end - 6:end]  # a "(YYYY)" citation is always six characters
    words = list(re.finditer(r"\S+", text[: end - 6]))
    for count in range(min(len(words), 8), 0, -1):
        if " ".join([word.group() for word in words[-count:]] + [year]).casefold() in answers_folded:
            return "answer", words[-count].start()
    return None, None


def _without_spans(text: str, spans: Sequence[tuple[int, int]]) -> str:
    """``text`` with the given (start, end) character ranges blanked out, keeping offsets."""
    for start, end in spans:
        text = text[:start] + " " * (end - start) + text[end:]
    return text


def check_explanation(movie_id: int, text: str, evidence: Evidence) -> dict:
    """Screen one explanation. Returns {movie_id, status, problems, verified}."""
    doc = evidence.shown[movie_id]
    problems: list[dict] = []
    verified = {"titles": 0, "ratings": 0, "genres": 0, "tags": 0, "decades": 0, "answer_references": 0}

    def flag(status: str, code: str, detail: str) -> None:
        problems.append({"status": status, "code": code, "detail": detail[:160]})

    if len(text) > MAX_TEXT:
        flag(INSUFFICIENT_EVIDENCE, "text_too_long", f"only the first {MAX_TEXT} characters were screened")
        text = text[:MAX_TEXT]
    answers_folded = "\n".join(evidence.answers).casefold()
    user_side = evidence.profile_text + "\n" + "\n".join(evidence.answers)
    user_folded = user_side.casefold()
    known = _known_titles(movie_id, evidence)

    # Titles: every "(YYYY)" citation must close a displayed title.
    cites = []  # (end offset, resolved)
    title_spans = []  # where cited titles sit: their words are names, not claims about genre
    for match in _YEAR_CITE.finditer(text):
        resolved, title_start = _resolve_cite(text, match.end(), known, answers_folded)
        cites.append((match.end(), resolved))
        if title_start is not None:
            title_spans.append((title_start, match.end()))
        if resolved is None:
            flag(UNSUPPORTED, "title_not_displayed", f"cited title not shown: ...{text[max(0, match.end() - 40):match.end()]}")
        elif isinstance(resolved, _Known) and resolved.side == "other":
            flag(UNSUPPORTED, "cites_other_candidate", f"{resolved.title} is another recommendation, not the user's history")
        else:
            verified["titles"] += 1

    # Ratings: "N/5" must follow a displayed user-side title with that rating.
    for match in _RATING.finditer(text):
        before = [(end, item) for end, item in cites if end <= match.start() and match.start() - end <= 120]
        source = before[-1][1] if before else None
        if not isinstance(source, _Known) or source.side != "user":
            if source == "answer":
                verified["ratings"] += 1  # a free-text answer's own rating cannot be compared
            else:
                flag(UNSUPPORTED, "rating_without_source", f"{match.group()} is not attached to a rated movie shown")
        elif abs(source.rating - float(match.group(1))) > 1e-9:
            flag(UNSUPPORTED, "rating_mismatch", f"{match.group()} but the profile shows {source.rating:g}/5 for {source.title}")
        else:
            verified["ratings"] += 1

    # Genres (outside quotes, which hold tags, and outside cited titles, which are names such as
    # "Mystery Men (1999)"): listed for this movie, else shown for the user, else not shown.
    prose = _without_spans(text, title_spans)
    unquoted = _QUOTED.sub("", prose)
    own_genres = set(doc.get("genres") or [])
    for genre, pattern in _GENRE_PATTERNS.items():
        if not pattern.search(unquoted):
            continue
        if genre in own_genres:
            verified["genres"] += 1
        elif pattern.search(user_side):
            flag(NEEDS_REVIEW, "genre_not_listed_for_movie", f"{genre} is shown for the user's history, not for this movie")
        else:
            flag(UNSUPPORTED, "genre_not_displayed", f"{genre} is not listed for this movie or shown for the user")

    # Quoted tags.
    own_tags = {tag.lower() for tag in (doc.get("tags") or [])}
    for tag in _QUOTED.findall(prose):  # a quoted word inside a cited title is part of the title
        lowered = tag.lower()
        if lowered in own_tags:
            verified["tags"] += 1
        elif lowered in user_folded:
            flag(NEEDS_REVIEW, "tag_not_listed_for_movie", f'"{tag}" is shown for the user\'s history, not for this movie')
        else:
            flag(UNSUPPORTED, "tag_not_displayed", f'"{tag}" is not a tag of this movie or shown for the user')

    # Decades.
    own_year = re.search(r"\((\d{4})\)\s*$", doc["title"])
    for decade in _DECADE.findall(text):
        in_answers = f"{decade}s" in answers_folded
        own = own_year is not None and own_year.group(1)[:3] == decade[:3]
        if own or in_answers:
            verified["decades"] += 1
        elif any(year.startswith(decade[:3]) for year in re.findall(r"\((\d{4})\)", evidence.profile_text)):
            flag(NEEDS_REVIEW, "decade_not_listed_for_movie", f"{decade}s appears only for the user's history")
        else:
            flag(UNSUPPORTED, "decade_not_displayed", f"{decade}s is not this movie's decade or in the user's answers")

    # References to what the user said.
    if _ANSWER_REF.search(text):
        if evidence.answers:
            verified["answer_references"] += 1
        else:
            flag(UNSUPPORTED, "answer_reference_without_answer", "refers to a clarification that did not happen")

    # Fields the data does not contain, unless the displayed text itself uses the word.
    shown_folded = (user_folded + "\n" + "\n".join(
        f"{d['title']} {' '.join(d.get('genres') or [])} {' '.join(d.get('tags') or [])}" for d in evidence.shown.values()
    )).casefold()
    for match in _UNAVAILABLE.finditer(text):
        if match.group().casefold() not in shown_folded:
            flag(INSUFFICIENT_EVIDENCE, "unavailable_field", f'"{match.group()}" is not a field in the catalog data')

    status = next((level for level in _SEVERITY if any(p["status"] == level for p in problems)), None)
    if status is None:
        status = PASSED_SCREEN if sum(verified.values()) else NO_CHECKABLE_CLAIM
    return {"movie_id": movie_id, "status": status, "problems": problems, "verified": verified}


def check_explanations(explanations: Sequence[Mapping], evidence: Evidence) -> list[dict]:
    return [check_explanation(item["movie_id"], item["text"], evidence) for item in explanations]
