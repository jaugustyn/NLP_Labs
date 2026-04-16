import os
import telebot
from dotenv import load_dotenv

from commands import register_handlers

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not TOKEN:
    raise ValueError("Missing TELEGRAM_BOT_TOKEN in .env file.")

bot = telebot.TeleBot(TOKEN)
register_handlers(bot)

if __name__ == "__main__":
    print("NLP Bot is starting...")
    bot.infinity_polling()
