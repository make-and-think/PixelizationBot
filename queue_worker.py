import time
import asyncio
import io
from datetime import datetime
from functools import partial
from collections import deque

from concurrent.futures import ProcessPoolExecutor

from telethon import events

from PIL import Image  # nah u to lazy to replace by wand
from torch.fx.experimental.unification.multipledispatch.dispatcher import source

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


class TaskQueue():
    def __int__(self, task_limit_per_user=config.SLOTS_QUANTITY):
        super().__init__()
        self.deque = deque()

        self.task_limit = task_limit_per_user
        self.item_form_source_count = {}

    def source_count_get(self, source):
        return self.item_form_source_count.get(source, 0)

    def append(self, source: any, item: any):
        task_count = self.item_form_source_count.get(source)

        if int(task_count) + 1 > self.task_limit:
            return None

        if not task_count:
            self.item_form_source_count.update({source: 1})
        else:
            self.item_form_source_count[source] += 1

        self.deque.append((source, item))
        return len(self.deque)

    def get(self):
        data = self.deque.popleft()
        source, item = data
        self.item_form_source_count[source] -= 1
        if self.item_form_source_count[source] <= 0:
            self.item_form_source_count.pop(source, None)
        return item

    def __len__(self):
        return len(self.deque)


class QueueWorkers:
    """QueueWorker to handle image queue"""

    def __init__(self, tbot, max_slots=config.SLOTS_QUANTITY):
        self.bot = tbot
        self.queue: deque[ImageToProcess] = deque()

        self._max_slots = max_slots
        self.task_queue = TaskQueue()

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

        self.last_time_image_processes = 0.0

        self.model_keep_alive_seconds = config.get("MODEL_KEEP_ALIVE_SECONDS")

        self.status_message = None

        self.workers_running = True

    def __del__(self):
        self.workers_running = False

    async def put_into_queue(self, input_messages: list[events.NewMessage.Event]):
        user_chat_id = input_messages[0].chat_id  # TODO make handle
        items_in_queue_from_chat = self.task_queue.source_count_get(user_chat_id)

        if items_in_queue_from_chat + len(input_messages) > self._max_slots:
            await self.bot.send_message(user_chat_id,
                                        f"To many images, now you have {items_in_queue_from_chat} slots in queue")
            if len(input_messages) > 1:
                await self.bot.send_message(user_chat_id, f"Now you try send {len(input_messages)} images")
            return

        task_list = [ImageToProcess(item_event, 0)
                     for item_event in input_messages]

        time_for_own_tasks = 0
        for task in task_list:
            time_for_own_tasks += task.predict_time_to_processes(self.compute_coefficient)

        time_in_queue = 0
        for _, task_in_queue in self.task_queue.deque:
            time_in_queue += task_in_queue.predict_time_to_processes(self.compute_coefficient)

        await self.bot.send_message(user_chat_id, f"""
        Time to handle you images ~{time_for_own_tasks}.\n
        Time before start process you image ~{time_in_queue}.\n
        You current position in queue: {len(self.task_queue.deque)}
""")

        for item in task_list:
            self.task_queue.append(user_chat_id, item)

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

            copy_instance_reverse = self.queue.copy()
            copy_instance_reverse.reverse()

            total_time_to_wait = 0

            for i, image_task in enumerate(copy_instance_reverse):
                if image_task.event.chat_id == current_chat_id:
                    images_in_queue = len(copy_instance_reverse) - i
                    for image_task_before in self.queue.copy():
                        total_time_to_wait += image_task_before.predict_time_to_processes(self.compute_coefficient)
                        if image_task.event.id == image_task_before.event.id:
                            break
                    status_message = f"Images in queue: {images_in_queue},estimated wait time: {total_time_to_wait:.2f} seconds"
                    await self._send_status_message(current_chat_id, status_message)
                    break

    async def _send_status_message(self, chat_id, message):
        if (self.user_queue_status.get(chat_id) is None) and (chat_id in self.user_queue_status):
            return

        if chat_id not in self.user_queue_status:
            self.user_queue_status.update({chat_id: None})
            self.user_queue_status.update({chat_id: await self.bot.send_message(chat_id, message)})
            return

        if self.user_queue_status[chat_id].message != message:
            try:
                await self.user_queue_status[chat_id].edit(message)
            except Exception as e:
                logger.debug(f"Error editing message: {e}")
                self.user_queue_status[chat_id] = await self.bot.send_message(chat_id, message)

        if (time.time() - self.user_queue_status[chat_id].date.timestamp()) > config.DELAY_STATUS:
            self.user_queue_status[chat_id] = await self.bot.send_message(chat_id, message)

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

    async def work_loop(self, worker_number: int):
        logger.info(f"Start Worker {worker_number}")
        while self.workers_running:

            if ((time.time() - self.last_time_image_processes)
                    > self.model_keep_alive_seconds
                    and (self.model_worker.G_A_net and
                         self.model_worker.alias_net)
                    and not len(self.task_queue.deque)
            ):
                logger.info("Unloading models due to inactivity.")
                self.model_worker.unload()

            if not len(self.task_queue.deque):
                await asyncio.sleep(1)
                continue

            if (self.model_worker.G_A_net and
                    self.model_worker.alias_net):
                logger.info("Load models after inactivity")
                self.model_worker.load()

            self.last_time_image_processes = time.time()

    async def worker_loop(self, number):
        logger.info(f"Start worker {number}")
        while True:
            if not len(self.queue):
                await asyncio.sleep(1)
                # if bot to long not work we unload model
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
