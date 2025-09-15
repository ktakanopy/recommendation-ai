import asyncio
import os
import logging

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

from big_boss.applications.graphs.onboarding_graph import OnboardingGraph
from big_boss.infrastructure.config.settings import BotSettings
from big_boss.infrastructure.llm.openai_client import LangChainOpenAILLM


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

settings = BotSettings()
llm = LangChainOpenAILLM(settings)
_graph = OnboardingGraph(settings, llm_port=llm)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("/start from user_id=%s", update.effective_user.id if update and update.effective_user else None)
    await update.message.reply_text("Welcome! Tell me a bit about you. For example: 'I'm 28 and a software engineer'.")


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    text = update.message.text or ""
    logger.info("message user_id=%s text=%s", user_id, text)
    try:
        reply = await _graph.handle_message(user_id, text)
    except Exception as e:
        logger.exception("graph error user_id=%s", user_id)
        reply = "Sorry, something went wrong. Please try again."
    logger.info("reply user_id=%s text=%s", user_id, reply)
    await update.message.reply_text(reply)


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("unhandled telegram error")


def main():
    logger.info("starting telegram bot")
    app = ApplicationBuilder().token(settings.TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), echo))
    app.add_error_handler(on_error)
    app.run_polling()


if __name__ == "__main__":
    main()
