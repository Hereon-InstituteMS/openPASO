"""A search that could not be made must not read as a web with nothing in it.

DuckDuckGo answers an EMPTY LIST when it throttles a machine rather than
raising, so "no results" was reported for a query that was never actually run.
An agent then concluded the literature has nothing on a standard benchmark and
worked from memory. Measured on a throttled machine: five queries in a row
returned nothing on all three backends, and the same queries returned hits
seconds later.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "langgraph_eval"))

import agent


class _FakeDDGS:
    """Stands in for the search client. `answers` is one list per call."""

    calls = 0
    answers: list = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def text(self, query, max_results=5, backend="auto"):
        type(self).calls += 1
        i = min(type(self).calls - 1, len(type(self).answers) - 1)
        return type(self).answers[i]


def _use(monkeypatch, answers):
    _FakeDDGS.calls, _FakeDDGS.answers = 0, answers
    monkeypatch.setattr(agent, "_SEARCH_CACHE", {}, raising=False)
    monkeypatch.setattr("duckduckgo_search.DDGS", _FakeDDGS, raising=False)
    monkeypatch.setattr(agent.time, "sleep", lambda s: None)


def test_a_throttled_search_says_it_could_not_search(monkeypatch):
    _use(monkeypatch, [[]])                       # every backend, every attempt: empty
    out = agent.web_search.invoke({"query": "Schaefer Turek cylinder benchmark", "max_results": 3})
    assert "could not search" in out
    assert "NOT as 'the web has nothing on this'" in out
    assert _FakeDDGS.calls == 9, "three backends, three attempts"


def test_a_later_attempt_still_counts(monkeypatch):
    hit = [{"title": "Benchmark Computations of Laminar Flow Around a Cylinder",
            "href": "https://example.org/turek", "body": "Cd, Cl and Strouhal for Re=100."}]
    _use(monkeypatch, [[], [], [], hit])          # blocked three times, then served
    out = agent.web_search.invoke({"query": "cylinder benchmark", "max_results": 3})
    assert "Benchmark Computations" in out and "could not search" not in out


def test_the_same_question_is_not_asked_twice(monkeypatch):
    hit = [{"title": "t", "href": "h", "body": "b"}]
    _use(monkeypatch, [hit])
    first = agent.web_search.invoke({"query": "Re=100 cylinder", "max_results": 3})
    before = _FakeDDGS.calls
    again = agent.web_search.invoke({"query": "  re=100 CYLINDER ", "max_results": 3})
    assert again == first and _FakeDDGS.calls == before, "a repeat costs no request"
