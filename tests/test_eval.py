from app.eval import aggregate, score


def test_score_requires_every_fact_and_accepts_alternatives():
    item = {"must_include": ["2 to 3|2-3", "4.6"]}
    assert score(item, {"answer": "OPEC: 2\u20133 mb/d vs 4.6 mb/d"})["correct"] is True
    partial = score(item, {"answer": "About 4.6 million barrels"})
    assert partial["correct"] is False and partial["fact_recall"] == 0.5


def test_unanswerable_is_correct_only_when_declined():
    item = {"answerable": False}
    assert score(item, {"answer": "x", "abstained": True})["correct"] is True
    assert score(item, {"answer": "The report does not mention hydrogen trucks."})["correct"] is True
    for declined in ("They do not state a rate.", "The excerpts do not say.", "It doesn't report one."):
        assert score(item, {"answer": declined})["correct"] is True
    assert score(item, {"answer": "About 2 million units."})["correct"] is False


def test_aggregate_separates_llm_decision_calls_from_generation():
    rows = [
        {
            "answerable": True,
            "correct": True,
            "fact_recall": 1.0,
            "abstained": False,
            "latency_s": 1.0,
            "retrieval_attempts": 1,
            "metrics": {"llm_calls": 3, "llm_calls.grade": 2, "llm_calls.generate": 1},
        }
    ]
    s = aggregate(rows)
    assert s["accuracy"] == 1.0 and s["decision_llm_calls"] == 2 and s["llm_calls"] == 3
