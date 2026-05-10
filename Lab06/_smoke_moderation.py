"""Offline smoke tests for the Lab06 moderation pipeline."""
import json
import uuid

from moderation.pipeline import moderate_content, policy_check
from moderation import actions, storage

RUN = uuid.uuid4().hex[:8]


def show(name, result):
    print(f"=== {name} ===")
    print(json.dumps(result, ensure_ascii=False, indent=2)[:1200])
    print()


clean = moderate_content(
    "Uwielbiam ten produkt, najlepszy zakup!",
    user_id=f"smoke_clean_{RUN}",
    username="SmokeClean",
)
show("clean approve", clean)

toxic = moderate_content(
    "Jesteś głupi i powinieneś się zabić",
    user_id=f"smoke_toxic_{RUN}",
    username="SmokeToxic",
)
show("toxic reject", toxic)

pii = moderate_content(
    "Mój email to john@example.com i numer +48 123 456 789",
    user_id=f"smoke_pii_{RUN}",
)
show("pii reject", pii)

dry = policy_check("Ci politykanci to wszystko złodzieje!")
show("dry policy check", dry)

show("history toxic", actions.get_user_moderation_history(f"smoke_toxic_{RUN}"))
show("similar", actions.find_similar_violations("glupi komentarz", limit=3))
show("feedback", actions.add_feedback(
    clean["content_id"],
    "APPROVE",
    "Smoke feedback example",
))
show("train_on_feedback", actions.train_on_feedback())
show("analytics", storage.analytics())

assert clean["action"] == "APPROVE"
assert toxic["action"] == "REJECT"
assert pii["action"] == "REJECT"
assert dry["action"] == "FLAG_FOR_REVIEW"
assert dry["content_id"].startswith("dry_")
print("Lab06 moderation smoke OK")
