import multiprocessing

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

from queue_worker import QueueWorkers

import asyncio
import logging
import os

from telethon import TelegramClient, events, functions
from telethon.tl.types import BotCommand, BotCommandScopeDefault, MessageMediaPhoto
from telethon.tl.functions.bots import SetBotCommandsRequest

from config.config import config

if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler('logs/bot.log'),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

bot = TelegramClient('bot', config.API_ID, config.API_HASH)

imageQueueWorker = QueueWorkers(bot)


async def set_bot_commands():
    commands = [
        BotCommand(command='start', description='Show available commands'),
        BotCommand(command='help', description='Get help on how to use the bot'),
    ]
    lang_code = 'en'

    await bot(SetBotCommandsRequest(scope=BotCommandScopeDefault(), lang_code=lang_code, commands=commands))


@bot.on(events.Album())
async def on_message(event):
    me = await bot.get_me()
    if event.sender_id and event.sender_id == me.id:
        return

    text = None
    event_list = []
    for message in event.messages:
        if text is None:
            text = message.text
        message.text = text
        event_list.append(message)

    await imageQueueWorker.put_into_queue(event_list)


@bot.on(events.NewMessage())
async def on_message(event):
    bot_me_obj = await bot.get_me()
    if event.grouped_id:  # Check if is album message
        return
    if event.sender_id and event.sender_id != bot_me_obj.id:
        if event.photo:
            await imageQueueWorker.put_into_queue([event])
        else:
            await event.reply('Please provide an image to pixelate. ')


@bot.on(events.NewMessage(pattern='/start'))
async def handle_start_command(event):
    await event.respond("Hello! Here are the available commands:\n/start - show available commands\n/help - get help.")


@bot.on(events.NewMessage(pattern='/help'))
async def help_message(event):
    help_text = (
        "I am a bot for pixelating images. Here’s what I can do:\n"
        "Send an image, and I will pixelate it for you.\n"
        "You can specify the pixel size by sending a number from 1 to 16 in the message along with the image."
    )
    await event.reply(help_text)


async def main():
    async with bot:
        logger.info("Starting the main bot loop")
        logger.info(f"Start botname :{await bot.get_me()}")
        await set_bot_commands()

        for image_work_task in imageQueueWorker.work_task_pool:
            logger.info("Put work task")
            bot.loop.create_task(image_work_task)

        bot.loop.create_task(imageQueueWorker.status_loop())

        logger.info("Starting the bot")  # Logging
        await bot.start(bot_token=config.API_TOKEN)
        await bot.run_until_disconnected()


if __name__ == '__main__':
    asyncio.run(main())
