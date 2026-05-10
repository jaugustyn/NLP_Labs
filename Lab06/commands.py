"""Telegram command registration entrypoint for the whole project."""

from lab1.commands import HELP_SECTION as LAB1_HELP_SECTION
from lab1.commands import register_handlers as register_lab1_handlers
from lab2.commands import HELP_SECTION as LAB2_HELP_SECTION
from lab2.commands import register_handlers as register_lab2_handlers
from lab3.commands import HELP_SECTION as LAB3_HELP_SECTION
from lab3.commands import register_handlers as register_lab3_handlers
from lab4.commands import HELP_SECTION as LAB4_HELP_SECTION
from lab4.commands import register_handlers as register_lab4_handlers
from lab5.commands import HELP_SECTION as LAB5_HELP_SECTION
from lab5.commands import register_handlers as register_lab5_handlers
from lab6.commands import HELP_SECTION as LAB6_HELP_SECTION
from lab6.commands import register_handlers as register_lab6_handlers


_LAB_HELP_SECTIONS = (
    LAB1_HELP_SECTION,
    LAB2_HELP_SECTION,
    LAB3_HELP_SECTION,
    LAB4_HELP_SECTION,
    LAB5_HELP_SECTION,
    LAB6_HELP_SECTION,
)


def _compose_help_text():
    sections = "\n\n".join(section.strip() for section in _LAB_HELP_SECTIONS)
    return "NLP Bot - Lab 1 + Lab 2 + Lab 3 + Lab 4 + Lab 5 + Lab 6\n\n" + sections


HELP_TEXT = _compose_help_text()


def register_handlers(bot):
    @bot.message_handler(commands=["start", "help"])
    def cmd_help(message):
        bot.reply_to(message, HELP_TEXT)

    register_lab1_handlers(bot)
    register_lab2_handlers(bot)
    register_lab3_handlers(bot)
    register_lab4_handlers(bot)
    register_lab5_handlers(bot)
    register_lab6_handlers(bot)


__all__ = ["HELP_TEXT", "register_handlers"]
