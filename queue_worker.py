import time
import asyncio
import io
from datetime import datetime
from functools import partial
from collections import deque

from concurrent.futures import ProcessPoolExecutor

from telethon import events

from PIL import Image  # nah u to lazy to replace by wand

from config.config import config, logger
from pixelization import PixelizationModel


class ImageToProcess:
    """Image task in queue"""

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
    """QueueWorker to handle image queue"""

    def __init__(self, tbot):
        self.bot = tbot
        self.queue: deque[ImageToProcess] = deque()

        self.user_queue_count = {}
        self.user_queue_status = {}

        self.model_worker = PixelizationModel()
        self.model_worker.load()

        self.process_pool = ProcessPoolExecutor(config.get("NUM_PROCESS"))
        self.work_task_pool = []

        for i in range(config.NUM_PROCESS):
            self.work_task_pool.append(self.worker_loop(i + 1))

        self.last_task_time = time.time()
        self.compute_coefficient = 1
        self.model_unload_timer = 0.0
        self.model_keep_alive_seconds = config.get("MODEL_KEEP_ALIVE_SECONDS")

        self.status_message = None

    async def put_into_queue(self, input_data: events.NewMessage.Event | list[events.NewMessage.Event]):
        task_list = []

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

        if (len(task_list) + current_chat_task_count) > config.SLOTS_QUANTITY:
            await self.bot.send_message(current_chat_id,
                                        f"To many images, now you have {config.SLOTS_QUANTITY - current_chat_task_count} slots in queue")
            return

        self.user_queue_count.update({current_chat_id: len(task_list) + current_chat_task_count})

        # Добаляем все задачи в очередь
        await self.update_status(current_chat_id)
        self.queue.extend(task_list)

    async def update_status(self, chat_id):
        """Update pos in queue"""
        for current_chat_id in self.user_queue_count.keys():
            total_time_to_wait = 0
            for i, image_task in enumerate(self.queue.copy()):
                total_time_to_wait += image_task.predict_time_to_processes(self.compute_coefficient)

                status_message = f"Images before you in queue: {i},\
                estimated wait time: {total_time_to_wait:.2f} seconds"

                await self._send_status_message(current_chat_id, status_message)
                if image_task.event.chat_id == current_chat_id:
                    break

    async def _send_status_message(self, chat_id, message):
        if (self.user_queue_status.get(chat_id) is None) and (chat_id in self.user_queue_status):
            return

        if chat_id not in self.user_queue_status:
            self.user_queue_status.update({chat_id: None})
            self.user_queue_status.update({chat_id: await self.bot.send_message(chat_id, message)})
            return
        if (time.time() - self.user_queue_status[chat_id].date.timestamp()) > config.DELAY_STATUS:
            # Если нет предыдущего сообщения или прошло больше 60 секунд, отправляем новое сообщение
            if self.user_queue_status[chat_id].text != message:
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
        output_image.name = f'output-pixai-image-{now.strftime("%d.%m.%Y-%H:%M:%S")}.png'
        processed_img.save(output_image, format='PNG')
        output_image.seek(0)

        return output_image

    async def worker_loop(self, number):
        logger.info(f"Start worker {number}")
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
                await self.bot.download_media(image_task.event.photo, file=input_image_bytes)
                output_image = await self.process_image(input_image_bytes, image_task.pixel_size)

                image_task.end_time = time.time()

                self.compute_coefficient = ((image_task.end_time - image_task.start_time) * image_task.pixel_size) / (
                        image_task.height * image_task.width)

                await self.bot.send_file(
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
                        self.user_queue_count.pop(current_chat_id)
                        await self.bot.send_message(current_chat_id, "All image processing complete!")
                        self.user_queue_status.pop(current_chat_id)
                    else:
                        self.user_queue_count.update({current_chat_id: current_chat_task_count - 1})

            except Exception as e:
                logger.error(f"Error when process image {e}")
                image_task.error = True
            self.last_task_time = time.time()
