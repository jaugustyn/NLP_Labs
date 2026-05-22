"""Telegram entrypoint for the Lab06 bot."""

import logging
import os

import telebot
from dotenv import load_dotenv

from commands import register_handlers

load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not TOKEN:
    raise ValueError("Missing TELEGRAM_BOT_TOKEN in .env file.")

bot = telebot.TeleBot(TOKEN)
register_handlers(bot)

if __name__ == "__main__":
    print("NLP Bot (Lab06) is starting...")
    bot.infinity_polling()
