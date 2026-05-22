"""Offline smoke test for the Lab 6 moderation pipeline."""

import os
import tempfile

from lab5 import tools as tools_mod
from lab6.moderation import actions, models, pipeline, storage


def _point_storage_to(temp_dir):
    storage.MODERATION_DATA_DIR = temp_dir
    storage.MODERATION_LOG = os.path.join(temp_dir, "moderation_log.csv")
    storage.USER_HISTORY = os.path.join(temp_dir, "user_moderation_history.csv")
    storage.FEEDBACK_LOG = os.path.join(temp_dir, "feedback_log.csv")
    storage.TRAIN_DATA = os.path.join(temp_dir, "train_data.csv")
    storage.ACTION_LOG = os.path.join(temp_dir, "moderation_actions.csv")
    storage.WATCHLIST = os.path.join(temp_dir, "watchlist.csv")


def _row_count(path):
    return len(storage._read_rows(path))


def main():
    with tempfile.TemporaryDirectory() as temp_dir:
        _point_storage_to(temp_dir)
        storage.ensure_storage()

        clean = pipeline.moderate_content(
            "Uwielbiam ten produkt, najlepszy zakup!",
            user_id="user_clean",
            username="Anna",
        )
        assert clean["action"] == "APPROVE"
        assert storage.get_moderation(clean["content_id"]) is not None

        pii = pipeline.moderate_content(
            "Mój numer to +48 123 456 789, email to john@example.com",
            user_id="user_pii",
        )
        assert pii["action"] == "REJECT"
        assert pii["reason"] == "personally_identifiable_information"
        assert pii["pii"]["has_pii"] is True

        card = models.detect_private_info("Karta testowa 4111 1111 1111 1111")
        assert any(e["type"] == "credit_card" for e in card["entities"])

        toxic = pipeline.moderate_content(
            "Jesteś beznadziejny i powinienes sie zabic",
            user_id="user_bad",
        )
        assert toxic["action"] == "REJECT"
        assert toxic.get("shadow_ban_result", {}).get("action") == "SHADOW_BAN"
        assert toxic.get("watchlist_result", {}).get("action") == "ADD_TO_WATCHLIST"

        similar = pipeline.moderate_content(
            "Powinienes sie zabic, to jest beznadziejne",
            user_id="user_bad_2",
        )
        assert similar["similar_cases"]

        before_dry_run = _row_count(storage.MODERATION_LOG)
        dry = pipeline.policy_check("Kliknij promocja http://x.test http://y.test")
        assert dry["content_id"].startswith("dry_")
        assert _row_count(storage.MODERATION_LOG) == before_dry_run

        political = models.classify_bielik_guard("Ci politykanci to złodzieje")
        assert "political_opinion" not in political["categories"]

        feedback = actions.add_feedback(clean["content_id"], "REJECT", "Manual fix")
        assert feedback["status"] == "recorded"
        train = actions.train_on_feedback()
        assert train["feedback_examples"] == 1
        assert train["training_examples"] == 1

        invalid = actions.add_feedback(clean["content_id"], "MAYBE", "bad")
        assert "error" in invalid

        actions.add_to_watchlist("user_bad", "duplicate check")
        actions.add_to_watchlist("user_bad", "duplicate check")
        watch_rows = [
            row for row in storage.list_watchlist()
            if row.get("user_id") == "user_bad"
        ]
        assert len(watch_rows) == 1

        tool_names = tools_mod.list_tool_names()
        for name in (
            "approve_content",
            "reject_content",
            "flag_for_human_review",
            "shadow_ban_user",
            "get_user_moderation_history",
            "find_similar_violations",
            "add_to_watchlist",
            "add_feedback",
            "train_on_feedback",
        ):
            assert name in tool_names

        similar_tool = tools_mod.call_tool(
            "find_similar_violations",
            {"text": "beznadziejny zabic", "limit": "bad"},
        )
        assert isinstance(similar_tool, list)

        analytics = storage.analytics()
        assert analytics["total"] == 4
        assert analytics["actions"]["REJECT"] >= 2

    print("Lab6 smoke OK")


if __name__ == "__main__":
    main()
