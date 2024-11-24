import multiprocessing
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

import uuid
import time
import asyncio
import logging
import os
import io
import tempfile
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import aiofiles
import aiofiles.os
from telethon import TelegramClient, events, functions
from telethon.tl.types import BotCommand, BotCommandScopeDefault
from telethon.tl.functions.bots import SetBotCommandsRequest
from pixelization import PixelizationModel
from PIL import Image  # nah u to lazy to replace by wand

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


class AsyncTempFile:
    def __init__(self, ext=''):
        self.fd = None
        self.name = uuid.uuid4().hex + ext

    async def __aenter__(self):
        self.fd = await aiofiles.open(self.name, 'wb+')
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.fd.close()
        await aiofiles.os.remove(self.name)

    async def write(self, data):
        await self.fd.write(data)


class Task:
    def __init__(self, event, queue_n):
        self.event = event
        self.queue_n = queue_n
        self.message = None
        self.pixel_size = 4
        self.start_ts = 0
        self.error = False

        if event.text:
            try:
                self.pixel_size = int(event.text)
                self.pixel_size = max(1, min(self.pixel_size, 16))
            except ValueError:
                pass

    async def update_message(self, offset=0):
        self.queue_n += offset

        if not processor.times_history:
            avg_time = 10
        else:
            avg_time = int(sum(processor.times_history) / len(processor.times_history))

        text = 'Please wait, your image is being processed... '
        if self.error:
            text += 'Something went wrong during processing. Please try again later.'
        elif self.queue_n <= 0:
            text += 'Done.'
        elif self.queue_n == 1:
            text += f'(estimated time: {avg_time} sec.)'
        else:
            text += f'(position in the queue: {self.queue_n}; estimated time: {self.queue_n * avg_time} sec.)'

        try:
            if not self.message:
                self.message = await self.event.reply(text)
                return
            await self.message.edit(text)
        except Exception as e:
            print(f'Error updating message: {e}')


class QueueProcessor:
    def __init__(self):
        self.queue = []
        self.qsize = 0
        self.times_history = []
        self.model_worker = PixelizationModel()
        self.model_worker.load()
        self.process_pool = ProcessPoolExecutor(max_workers=config.NUM_PROCESS)
        self.last_task_time = time.time()

    async def process_image(self, image_bytes, pixel_size):
        # Выполняем обработку изображения в отдельном процессе
        loop = asyncio.get_event_loop()
        image_bytes.seek(0)
        original_img = Image.open(image_bytes)

        process_func = partial(
            self.model_worker.pixelize,
            original_img,
            pixel_size,
            upscale_after=True,
            copy_hue=True,
            copy_sat=True
        )
        #
        processed_img = await loop.run_in_executor(
            self.process_pool,
            process_func
        )

        #processed_img = self.model_worker.pixelize(original_img,pixel_size,upscale_after=True,copy_hue=True,copy_sat=True)


        output_image = io.BytesIO()
        processed_img.save(output_image, format='PNG')
        output_image.seek(0)
        output_image.name = 'processed_image.png'
        return output_image

    async def add_task(self, event):
        self.qsize += 1
        task = Task(event=event, queue_n=self.qsize)
        self.queue.append(task)
        await task.update_message()
        logger.info(f"Task added for processing image: {event.photo}")  # Logging

    async def loop(self):
        while True:
            if not self.queue:
                # Проверяем, прошло ли время удержания модели в памяти
                if self.model_worker.G_A_net is not None and time.time() - self.last_task_time > config.MODEL_KEEP_ALIVE_SECONDS:
                    self.model_worker.unload()
                await asyncio.sleep(1)
                continue

            logger.info("Start task")
            task = self.queue.pop(0)
            task.start_ts = time.time()
            self.last_task_time = task.start_ts  # Обновляем время последней задачи

            # Загружаем модель, если она была выгружена
            if self.model_worker.G_A_net is None:
                self.model_worker.load()

            try:
                input_image_bytes = io.BytesIO()
                logger.info(
                    f"Downloading image: ID={task.event.photo.id}, \
                    Access Hash={task.event.photo.access_hash}, \
                    Date={task.event.photo.date}, \
                    Sizes={[(size.type, size.w, size.h) for size in task.event.photo.sizes]}")

                await bot.download_media(task.event.photo, file=input_image_bytes)

                output_image = await self.process_image(input_image_bytes, task.pixel_size)

                await bot.send_file(
                    task.event.chat_id,
                    output_image,
                    filename=output_image.name,
                    force_document=True
                )
            except Exception as e:
                logger.error(f'Error processing task: {e}')
                task.error = True
            finally:
                self.times_history.append(time.time() - task.start_ts)
                if len(self.times_history) > 10:
                    self.times_history.pop(0)

                self.qsize -= 1
                await task.update_message(-1)
                for remaining_task in self.queue:
                    await remaining_task.update_message(-1)


processor = QueueProcessor()


# async def set_bot_commands():
#     commands = [
#         BotCommand(command='start', description='Show available commands'),
#         BotCommand(command='help', description='Get help on how to use the bot'),
#     ]
#     lang_code = 'en'
#
#     await bot(SetBotCommandsRequest(scope=BotCommandScopeDefault(), lang_code=lang_code, commands=commands))


@bot.on(events.NewMessage())
async def on_message(event):
    me = await bot.get_me()

    # Проверка, что сообщение отправлено пользователем, а не ботом
    if event.sender_id and event.sender_id != me.id:  # Используем sender_id
        if not event.photo:
            await event.reply('Please provide an image to pixelate.')
            return
        await processor.add_task(event)
    else:
        logger.info("Message from bot ignored.")  # Логирование игнорируемого сообщения


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
        await bot.loop.create_task(processor.loop())
        bot.start(bot_token=config.API_TOKEN)
        logger.info("Starting the bot")  # Logging
        await bot.run_until_disconnected()


if __name__ == '__main__':
    asyncio.run(main())
