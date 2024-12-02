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
from torch.multiprocessing import Pool
from datetime import datetime
from functools import partial
from collections import deque, Counter

import aiofiles
import aiofiles.os
from telethon import TelegramClient, events, functions
from telethon.tl.custom import Message
from telethon.tl.types import BotCommand, BotCommandScopeDefault, MessageMediaPhoto
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


class ImageToProcess:
    def __init__(self, event: events.NewMessage.Event, queue_pos, compute_coefficient=1):
        self.start_time = None
        self.end_time = None

        size = event.photo.sizes[-1]
        self.height = size.h
        self.width = size.w
        self.pixel_size = 4
        self.compute_coefficient = compute_coefficient

        self.event = event
        self.current_queue_pos = queue_pos
        self.relpy_message = None
        self.error = False

        if event.text:
            try:
                print(self.current_queue_pos, event.text)
                self.pixel_size = int(event.text)
                self.pixel_size = max(1, min(self.pixel_size, 16))
            except ValueError:
                pass

    def predict_time_to_processes(self, compute_coefficient):
        return compute_coefficient * ((self.height * self.width) / self.pixel_size)

    def real_time_to_processes(self):
        if self.start_time is None or self.end_time is None:
            logger.error("Start time or end time is not set.")
            return None
        return self.end_time - self.start_time

    def change_queue_pos(self, current_pos: int):
        self.current_queue_pos = current_pos


class QueueWorkers:
    def __init__(self):
        self.queue: deque[ImageToProcess] = deque()

        self.user_queue_count = {}
        self.user_queue_status = {}

        self.model_worker = PixelizationModel()
        self.model_worker.load()

        self.process_pool = ProcessPoolExecutor(config.get("NUM_PROCESS"))
        self.work_task_pool = []

        for _ in range(config.NUM_PROCESS):
            self.work_task_pool.append(self.worker_loop())

        self.last_task_time = time.time()
        self.compute_coefficient = 1
        self.model_unload_timer = 0.0
        self.model_keep_alive_seconds = config.get("MODEL_KEEP_ALIVE_SECONDS")

        self.status_message = None

    async def put_into_queue(self, input_data: events.NewMessage.Event | list[events.NewMessage.Event]):
        task_list = []
        current_chat_id = None

        if isinstance(input_data, list):
            current_chat_id = input_data[0].chat_id
            for event in input_data:
                logger.info(f"Task added for processing image: {event.photo}")
                task_list.append(
                    ImageToProcess(event, len(self.queue) + 1, compute_coefficient=self.compute_coefficient))
        else:
            current_chat_id = input_data.chat_id
            if input_data.photo:  # Проверяем, есть ли фото в сообщении
                logger.info(f"Task added for processing image: {input_data.photo}")
                task_list.append(
                    ImageToProcess(input_data, len(self.queue) + 1, compute_coefficient=self.compute_coefficient))

        if current_chat_id in self.user_queue_count:
            current_chat_task_count = self.user_queue_count.get(current_chat_id)
        else:
            current_chat_task_count = 0

        if (len(task_list) + current_chat_task_count) > 10:
            await bot.send_message(current_chat_id,
                                   f"To many images, now you have {10 - current_chat_task_count} slots in queue")
            return

        self.user_queue_count.update({current_chat_id: len(task_list) + current_chat_task_count})
        print('put_into_queue', self.user_queue_count)

        # Добаляем все задачи в очередь
        await self.update_status(current_chat_id)
        self.queue.extend(task_list)

    async def update_status(self, chat_id):
        """Update pos in queue"""
        print(self.user_queue_count.keys())
        for chat_id in self.user_queue_count.keys():
            total_time_to_wait = 0
            for i, image_task in enumerate(self.queue.copy()):
                print(i, image_task)
                total_time_to_wait += image_task.predict_time_to_processes(self.compute_coefficient)
                status_message = f"Images in queue: {i}, estimated wait time: {total_time_to_wait:.2f} seconds"
                await self._send_status_message(chat_id, status_message)
                if image_task.event.chat_id == chat_id:
                    break

    async def _send_status_message(self, chat_id, message):
        if not chat_id in self.user_queue_status:
            self.user_queue_status[chat_id] = None
            self.user_queue_status[chat_id] = await bot.send_message(chat_id, message)

        if self.user_queue_status[chat_id] is None:
            return

        if (time.time() - self.user_queue_status[chat_id].date.timestamp()) > 60:
            # Если нет предыдущего сообщения или прошло больше 60 секунд, отправляем новое сообщение
            if not self.user_queue_status[chat_id].text != message:
                await self.user_queue_status[chat_id].edit(message)

    def _take_image_task(self):
        selected = self.queue.popleft()
        for i, image_task in enumerate(self.queue):
            image_task.change_queue_pos(i)
        return selected

    async def process_image(self, image_bytes: io.BytesIO, pixel_size: int):
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

        loop = asyncio.get_event_loop()
        processed_img = await loop.run_in_executor(
            self.process_pool,
            process_func
        )

        output_image = io.BytesIO()
        now = datetime.now()
        output_image.name = f'output-image-{now.strftime("%d.%m.%Y-%H:%M:%S")}.png'
        processed_img.save(output_image, format='PNG')
        output_image.seek(0)

        return output_image

    async def worker_loop(self):
        logger.info(f"Start worker {id(self)}")
        while True:
            if not len(self.queue):
                await asyncio.sleep(1)
                if self.model_unload_timer and (time.time() - self.model_unload_timer > self.model_keep_alive_seconds):
                    logger.info("Unloading models due to inactivity.")
                    self.model_worker.unload()
                    self.model_unload_timer = None
                continue

            logger.info("Start process image")
            self.model_unload_timer = time.time()

            if self.model_worker.G_A_net is None:
                logger.info("Loading models...")
                self.model_worker.load()

            image_task = self._take_image_task()
            await self.update_status(image_task.event.chat_id)
            image_task.change_queue_pos(-1)

            image_task.start_time = time.time()

            input_image_bytes = io.BytesIO()
            logger.info(
                f"Downloading image: ID={image_task.event.photo.id}, \
                                Access Hash={image_task.event.photo.access_hash}, \
                                Date={image_task.event.photo.date}, \
                                Sizes={[(size.type, size.w, size.h) for size in image_task.event.photo.sizes]}")

            try:
                await bot.download_media(image_task.event.photo, file=input_image_bytes)
                output_image = await self.process_image(input_image_bytes, image_task.pixel_size)

                image_task.end_time = time.time()

                self.compute_coefficient = ((image_task.end_time - image_task.start_time) * image_task.pixel_size) / (
                        image_task.height * image_task.width)

                await bot.send_file(
                    image_task.event.chat_id,
                    output_image,
                    filename=output_image.name,
                    force_document=True
                )

                image_task.change_queue_pos(-2)
                current_chat_id = image_task.event.chat_id
                if current_chat_id in self.user_queue_count:
                    current_chat_task_count = self.user_queue_count.get(current_chat_id)
                    if (current_chat_task_count - 1) == 0:
                        del self.user_queue_count[current_chat_id]
                        await bot.send_message(current_chat_id, "All image processing complete!")
                        del self.user_queue_status[current_chat_id]
                    else:
                        self.user_queue_count.update({current_chat_id: current_chat_task_count - 1})

            except Exception as e:
                logger.error(f"Error when process image {e}")
                image_task.error = True
            self.last_task_time = time.time()


processor = QueueWorkers()


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
    await processor.put_into_queue(event_list)


@bot.on(events.NewMessage())
async def on_message(event):
    me = await bot.get_me()
    if event.grouped_id:
        return
    # Проверка, что сообщение отправлено пользователем, а не ботом
    if event.sender_id and event.sender_id != me.id:

        if event.photo:  # Проверяем, есть ли фото в сообщении
            await processor.put_into_queue(event)  # Помещаем одно изображение в очередь
        else:
            await event.reply('Please provide an image to pixelate.')
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
        await set_bot_commands()
        for image_work_task in processor.work_task_pool:
            logger.info("Put work task")
            bot.loop.create_task(image_work_task)

        logger.info("Starting the bot")  # Logging
        await bot.start(bot_token=config.API_TOKEN)
        await bot.run_until_disconnected()


if __name__ == '__main__':
    asyncio.run(main())
