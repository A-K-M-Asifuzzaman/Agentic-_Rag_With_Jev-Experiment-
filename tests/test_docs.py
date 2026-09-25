from pathlib import Path

from app.agent.graph import mermaid

DOC = (Path(__file__).parents[1] / "docs" / "ARCHITECTURE.md").read_text()


def test_documented_diagrams_match_compiled_graphs():
    for pipeline in ("hybrid", "standard"):
        assert mermaid(pipeline).strip() in DOC, f"regenerate docs diagram: ajev graph --pipeline {pipeline}"
