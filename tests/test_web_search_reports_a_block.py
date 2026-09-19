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
    """Stand in for whichever search package is installed.

    The tool imports duckduckgo_search and falls back to ddgs, so a test that
    patches one of them by name passes or fails according to what happens to be
    installed. Both names are provided here."""
    import types

    _FakeDDGS.calls, _FakeDDGS.answers = 0, answers
    monkeypatch.setattr(agent, "_SEARCH_CACHE", {}, raising=False)
    monkeypatch.setattr(agent, "_SEARCH_BLOCKED", {}, raising=False)
    for name in ("duckduckgo_search", "ddgs"):
        module = types.ModuleType(name)
        module.DDGS = _FakeDDGS
        monkeypatch.setitem(sys.modules, name, module)
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
    again = agent.web_search.invoke({"query": "  Re=100 cylinder ", "max_results": 3})
    assert again == first and _FakeDDGS.calls == before, "a repeat costs no request"


def test_a_differently_spelled_question_is_a_different_question(monkeypatch):
    """What is remembered must be what was sent: a search engine's results are
    not case-blind, so a lowercased key must not answer for another spelling."""
    hit = [{"title": "t", "href": "h", "body": "b"}]
    _use(monkeypatch, [hit])
    agent.web_search.invoke({"query": "Re=100 cylinder", "max_results": 3})
    before = _FakeDDGS.calls
    agent.web_search.invoke({"query": "re=100 CYLINDER", "max_results": 3})
    assert _FakeDDGS.calls > before, "a different spelling is asked, not served from the cache"


def test_the_cache_does_not_grow_without_limit(monkeypatch):
    """A web interface keeps this process up for days, and the key is whatever
    anyone searched for."""
    hit = [{"title": "t", "href": "h", "body": "b"}]
    _use(monkeypatch, [hit])
    for i in range(agent._SEARCH_CACHE_MAX + 20):
        agent.web_search.invoke({"query": f"question {i}", "max_results": 3})
    assert len(agent._SEARCH_CACHE) <= agent._SEARCH_CACHE_MAX
    # and what it still holds is the most recent, not the first
    assert ("", f"question {agent._SEARCH_CACHE_MAX + 19}", 3) in agent._SEARCH_CACHE
    assert ("", "question 0", 3) not in agent._SEARCH_CACHE


def test_one_run_is_not_served_another_run_s_search(monkeypatch):
    """A web interface serves many runs from one process. Without a scope, a run
    could be handed snippets another run fetched, and its transcript would show
    results it never asked for."""
    hit = [{"title": "t", "href": "h", "body": "b"}]
    _use(monkeypatch, [hit])
    token = agent.SEARCH_SCOPE.set("run-a")
    agent.web_search.invoke({"query": "cylinder benchmark", "max_results": 3})
    after_a = _FakeDDGS.calls
    agent.SEARCH_SCOPE.reset(token)
    agent.SEARCH_SCOPE.set("run-b")
    agent.web_search.invoke({"query": "cylinder benchmark", "max_results": 3})
    assert _FakeDDGS.calls > after_a, "the second run asks for itself"


def test_a_query_just_refused_is_not_retried_at_once(monkeypatch):
    """Nine requests and five seconds of pauses per repeat, and repetition is
    what causes the throttling."""
    _use(monkeypatch, [[]])
    first = agent.web_search.invoke({"query": "blocked question", "max_results": 3})
    spent = _FakeDDGS.calls
    again = agent.web_search.invoke({"query": "blocked question", "max_results": 3})
    assert "could not search" in first and again.startswith(agent._BLOCKED_MESSAGE[:40])
    assert _FakeDDGS.calls == spent, "the repeat costs nothing"
    # and it is remembered only briefly, so a provider that recovers is reachable
    later = agent.time.time() + 120                  # the clock the tool reads
    monkeypatch.setattr(agent.time, "time", lambda: later)
    _FakeDDGS.answers = [[{"title": "t", "href": "h", "body": "b"}]]
    assert "could not search" not in agent.web_search.invoke({"query": "blocked question", "max_results": 3})


def test_a_broken_connection_is_not_reported_as_a_block(monkeypatch):
    """Every attempt raising is a failure to reach the provider, which is not
    the provider answering "nothing" — and it must not hold the query back for
    a minute, because the next attempt may well work."""
    class _Broken(_FakeDDGS):
        def text(self, query, max_results=5, backend="auto"):
            type(self).calls += 1
            raise OSError("connection reset by peer")

    import types
    _FakeDDGS.calls = 0
    monkeypatch.setattr(agent, "_SEARCH_CACHE", {}, raising=False)
    monkeypatch.setattr(agent, "_SEARCH_BLOCKED", {}, raising=False)
    for name in ("duckduckgo_search", "ddgs"):
        module = types.ModuleType(name)
        module.DDGS = _Broken
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(agent.time, "sleep", lambda s: None)

    out = agent.web_search.invoke({"query": "anything", "max_results": 3})
    assert "could not be made" in out and "connection reset" in out
    assert "could not search" not in out, "a broken connection is not a throttle"
    assert agent._SEARCH_BLOCKED == {}, "and it is not held against the query"
