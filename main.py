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
from collections import deque

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
        return self.end_time - self.start_time

    def change_queue_pos(self, current_pos: int):
        self.current_queue_pos = current_pos

    async def update_status_message(self, time_to_wait=0):
        status_text = 'Please wait, your image is being processed...\n'
        if self.error:
            status_text = 'Something went wrong during processing. Please try again later.'
        elif self.current_queue_pos == -1:
            self.start_time = time.time()
            status_text += f'(Estimated time: {time_to_wait:.2f} sec.)'
        elif self.current_queue_pos <= -1:
            status_text += 'Done.'
            self.end_time = time.time()
        else:
            status_text += f'(Position in the queue: {self.current_queue_pos}; Estimated time: {time_to_wait:.2f} sec.)'

        try:
            if not self.relpy_message:
                self.relpy_message = await self.event.reply(status_text)
                return
            await self.relpy_message.edit(status_text)
        except Exception as e:
            logger.info(f'Error updating message: {e}')
        return self.current_queue_pos


class QueueWorkers:
    def __init__(self):
        self.queue: deque[ImageToProcess] = deque()

        self.times_history = []
        self.model_worker = PixelizationModel()
        
        #if self.model_worker.G_A_net is None:
        #       self.model_worker.load()

        self.process_pool = ProcessPoolExecutor(config.get("NUM_PROCESS"))
        self.work_task_pool = []
        for _ in range(config.NUM_PROCESS):
            self.work_task_pool.append(self.worker_loop())
        self.last_task_time = time.time()

        self.compute_coefficient = 1
        self.model_unload_timer = None  # Таймер для выгрузки модели
        self.model_keep_alive_seconds = config.get("MODEL_KEEP_ALIVE_SECONDS")

    def put_into_queue(self, input_data: events.NewMessage.Event | list[events.NewMessage.Event]):
        task_list = []
        if isinstance(input_data, list):
            for event in input_data:
                logger.info(f"Task added for processing image: {event.photo}")
                task_list.append(
                    ImageToProcess(event, len(self.queue) + 1, compute_coefficient=self.compute_coefficient))
        else:
            task_list = [ImageToProcess(input_data, len(self.queue) + 1, compute_coefficient=self.compute_coefficient)]
        self.queue.extend(task_list)

    async def update_status(self):
        time_to_wait = 0
        for image_task in self.queue.copy():
            time_to_wait += image_task.predict_time_to_processes(self.compute_coefficient)
            #await image_task.update_status_message(time_to_wait)

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

            # Сбрасываем таймер выгрузки модели, так как начинается новая задача
            self.model_unload_timer = time.time()

            # Загружаем модель, если она была выгружена
            if self.model_worker.G_A_net is None:
                logger.info("Loading models...")
                self.model_worker.load()

            image_task = self._take_image_task()
            image_task.change_queue_pos(-1)

            # Проверяем, что модель загружена перед обработкой
            if self.model_worker.G_A_net is None:
                logger.error("Model is not loaded properly.")
                continue

            time_to_wait = image_task.predict_time_to_processes(self.compute_coefficient)
            await self.update_status()
            await image_task.update_status_message(time_to_wait)

            input_image_bytes = io.BytesIO()
            logger.info(
                f"Downloading image: ID={image_task.event.photo.id}, \
                                Access Hash={image_task.event.photo.access_hash}, \
                                Date={image_task.event.photo.date}, \
                                Sizes={[(size.type, size.w, size.h) for size in image_task.event.photo.sizes]}")

            try:
                await bot.download_media(image_task.event.photo, file=input_image_bytes)
                output_image = await self.process_image(input_image_bytes, image_task.pixel_size)
                self.compute_coefficient = ((time.time() - image_task.start_time) * image_task.pixel_size) / (
                        image_task.height * image_task.width)
                await bot.send_file(
                    image_task.event.chat_id,
                    output_image,
                    filename=output_image.name,
                    force_document=True
                )

                image_task.change_queue_pos(-2)
                await image_task.update_status_message()

            except Exception as e:
                logger.error(f"Error when process image {e}")
                image_task.error = True
                await image_task.update_status_message()
            self.last_task_time = time.time()  # Обновляем время последней задачи

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

    processor.put_into_queue(event_list)


@bot.on(events.NewMessage())
async def on_message(event):
    me = await bot.get_me()

    # Проверка, что сообщение отправлено пользователем, а не ботом
    if event.sender_id and event.sender_id != me.id:
        if event.grouped_id:
            return
        if not event.photo:
            await event.reply('Please provide an image to pixelate.')
            return
        processor.put_into_queue(event)
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
