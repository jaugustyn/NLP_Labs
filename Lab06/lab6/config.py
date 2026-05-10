"""Configuration specific to Lab06 moderation."""
import os

from config import BASE_DIR


MODERATION_DATA_DIR = os.path.join(BASE_DIR, "moderation_data")
MODERATION_DEFAULT_MODERATOR_ID = "bot"

