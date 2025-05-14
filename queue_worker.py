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


class ImageTaskToProcess:
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

    def __init__(self, task_limit_per_user=config.SLOTS_QUANTITY):
        self.deque = deque()
        self.task_limit = task_limit_per_user
        self.item_form_source_count = {}

    def source_count_get(self, source):
        return self.item_form_source_count.get(source, 0)

    def append_task(self, source: any, item: any):
        task_count = self.item_form_source_count.get(source, 0)

        if int(task_count) + 1 > self.task_limit:
            return None

        if not task_count:
            self.item_form_source_count.update({source: 1})
        else:
            self.item_form_source_count[source] += 1

        self.deque.append((source, item))
        return len(self.deque)

    def get_task(self):
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
        self.queue: deque[ImageTaskToProcess] = deque()

        self._max_slots = max_slots
        self.task_queue = TaskQueue()

        self.user_queue_count = {}
        self.user_queue_status = {}

        self.model_worker = PixelizationModel()
        self.model_worker.load()

        self.process_pool = ProcessPoolExecutor(config.get("NUM_PROCESS"))
        self.work_task_pool = []

        for i in range(config.NUM_PROCESS):
            self.work_task_pool.append(self.work_loop(i + 1))

        self.last_task_time = time.time()
        self.compute_coefficient = 1
        self.model_unload_timer = 0.0

        self.last_time_image_processes = 0.0

        self.model_keep_alive_seconds = config.get("MODEL_KEEP_ALIVE_SECONDS")

        self.status_message = None

        self.workers_running = True

        # make test run
        start_test_run = time.time()
        with Image.open("images/test_image.png") as image:
            self.model_worker.pixelize(image, 6)
        end_test_run = time.time()
        end_test_run += 1
        self.compute_coefficient = (((end_test_run - start_test_run) * 6) / (
                image.height * image.width))

    def __del__(self):
        self.workers_running = False
        self.work_task_pool = []

    async def put_into_queue(self, input_messages: list[events.NewMessage.Event]):
        print(input_messages)
        user_chat_id = input_messages[0].chat_id  # TODO make handle
        items_in_queue_from_chat = self.task_queue.source_count_get(user_chat_id)

        if items_in_queue_from_chat + len(input_messages) > self._max_slots:
            status_message = f"""To many images, now you have {items_in_queue_from_chat} pictures in queue"
Now you try send {len(input_messages)} images
            """
            await self.bot.send_message(user_chat_id, status_message)
            return

        task_list = [ImageTaskToProcess(item_event, 0)
                     for item_event in input_messages]

        time_for_own_tasks = 0
        for task in task_list:
            time_for_own_tasks += task.predict_time_to_processes(self.compute_coefficient)

        time_in_queue = 0
        for _, task_in_queue in self.task_queue.deque:
            time_in_queue += task_in_queue.predict_time_to_processes(self.compute_coefficient)

        status_message = f"""Time to processing you new images ~{time_for_own_tasks:.2f}.
Time before start process you image ~{time_in_queue:.2f}.
You current position in queue: {len(self.task_queue.deque)}
"""
        await self.bot.send_message(user_chat_id, status_message)

        for item in task_list:
            print(user_chat_id, item, task_list)
            self.task_queue.append_task(user_chat_id, item)

    async def status_loop(self):
        while self.workers_running:
            await self.send_current_status()
            await asyncio.sleep(120)  # TODO use config file

    async def send_current_status(self):
        last_user_chat_id = None
        count_items = 0
        time_to_process_items = 0
        position_in_queue = 0

        if not len(self.task_queue.deque):
            return

        for user_chat_id, image_task in self.task_queue.deque:
            if last_user_chat_id is None:
                last_user_chat_id = user_chat_id

            time_to_process_items += image_task.predict_time_to_processes(self.compute_coefficient)
            count_items += 1

            if last_user_chat_id != user_chat_id:
                user_position_in_queue = 0 if (position_in_queue - count_items) < 0 else position_in_queue - count_items

                status_message = f"""You position in queue: {user_position_in_queue}
Time to processed you images (with wait time in queue): ~{time_to_process_items:.2f}"""

                await self.bot.send_message(last_user_chat_id, status_message)
                count_items = 0

            last_user_chat_id = user_chat_id
            position_in_queue += 1

        user_position_in_queue = 0 if (position_in_queue - count_items) < 0 else position_in_queue - count_items
        status_message = f"""You position in queue: {user_position_in_queue}
Time to processed you images (with wait time in queue): ~{time_to_process_items:.2f}"""
        await self.bot.send_message(last_user_chat_id, status_message)

    async def process_image(self, image_bytes: io.BytesIO, pixel_size: int):
        # TODO make try handle
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
            # Work loop cond
            if (((time.time() - self.last_time_image_processes)
                 > self.model_keep_alive_seconds)
                    and (self.model_worker.G_A_net is not None)
                    and (self.model_worker.alias_net is not None)
                    and len(self.task_queue.deque) == 0
            ):
                logger.info("Unloading models due to inactivity.")
                self.model_worker.unload()

            if len(self.task_queue.deque) == 0:
                await asyncio.sleep(1)
                continue

            if ((self.model_worker.G_A_net is None)
                    and (self.model_worker.alias_net is None)
            ):
                logger.info("Load models after inactivity")
                self.model_worker.load()

            self.last_time_image_processes = time.time()

            image_task = self.task_queue.get_task()

            image_task.start_time = time.time()  # Start timer of calc

            downloaded_image_bytes = await self._download_image(image_task)
            logger.info(f"Start processing image: ID={image_task.event.photo.id}")
            output_image = await self.process_image(downloaded_image_bytes, image_task.pixel_size)

            logger.info(f"Send processed image: ID={image_task.event.photo.id} ")

            await self.bot.send_file(
                image_task.event.chat_id,
                output_image,
                filename=output_image.name,
                force_document=True
            )

            image_task.end_time = time.time()  # End timer of calc

            self.compute_coefficient = (((image_task.end_time - image_task.start_time) * image_task.pixel_size) / (
                    image_task.height * image_task.width))

    async def _download_image(self, image_task: events.NewMessage.Event):
        input_image_bytes = io.BytesIO()
        logger.info(
            f"Downloading image: ID={image_task.event.photo.id}, \
                                        Access Hash={image_task.event.photo.access_hash}, \
                                        Date={image_task.event.photo.date}, \
                                        Sizes={[(size.type, size.w, size.h) for size in image_task.event.photo.sizes]}")
        try:
            await self.bot.download_media(image_task.event.photo, file=input_image_bytes)
        except Exception as error_message:
            logger.error(f"Error when download image {error_message}")

        return input_image_bytes
