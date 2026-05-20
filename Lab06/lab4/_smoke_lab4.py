"""Offline smoke test for Lab 4 core behavior."""

import os
import tempfile

from lab4 import knowledge_graph
from lab4 import language_detect
from lab4 import nel
from lab4 import ner
from lab4 import ned
from lab4 import translation


def main():
    assert language_detect.detect_language("Warszawa to stolica Polski") == "pl"
    assert language_detect.detect_language("Hello world") == "en"

    entities = ner.extract_entities(
        "Elon Musk founded SpaceX in Austin in 2002.",
        method="regex",
        lang="en",
    )
    entity_texts = {entity["text"] for entity in entities}
    assert "Elon Musk" in entity_texts
    assert "SpaceX" in entity_texts

    local = nel.search_local_kb("Elon Musk", lang="en")
    assert local and local[0]["qid"] == "Q317521"

    old_search_candidates = ned.search_candidates
    try:
        ned.search_candidates = lambda name, lang="en", limit=5: nel.search_local_kb(
            name,
            lang=lang,
        )
        ranked = ned.disambiguate(
            "Elon Musk",
            "Elon Musk founded SpaceX and xAI.",
            lang="en",
        )
        assert ranked and ranked[0]["qid"] == "Q317521"
    finally:
        ned.search_candidates = old_search_candidates

    assert translation.validate_pair("en", "pl")
    assert not translation.validate_pair("pl", "es")

    with tempfile.TemporaryDirectory() as temp_dir:
        old_plots_dir = knowledge_graph.PLOTS_DIR
        old_search_wikidata = nel.search_wikidata
        try:
            knowledge_graph.PLOTS_DIR = temp_dir
            nel.search_wikidata = lambda name, lang="en", limit=5: []
            linked = nel.link_entities(entities, lang="en")
            graph = knowledge_graph.build_graph([linked])
            stats = knowledge_graph.graph_stats(graph)
            assert stats["nodes"] >= 2
            path = knowledge_graph.plot_graph(graph)
            assert path and os.path.exists(path)
        finally:
            knowledge_graph.PLOTS_DIR = old_plots_dir
            nel.search_wikidata = old_search_wikidata

    print("Lab4 smoke OK")


if __name__ == "__main__":
    main()
