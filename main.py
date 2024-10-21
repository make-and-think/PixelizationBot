import uuid
import time
import asyncio

import aiofiles
import aiofiles.os
from telethon import TelegramClient
from telethon.events import NewMessage

from config import config

class AsyncTempFile:
  def __init__(self, ext=''):
    self.fd = None
    self.name = uuid.uuid4().hex + ext

  async def __aenter__(self):
    self.fd = await aiofiles.open(self.name, 'wb+')

    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    await self.fd.close()
    await aiofiles.os.unlink(self.name)

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
        self.pixel_size = max(self.pixel_size, 1)
        self.pixel_size = min(self.pixel_size, 16)
      except:
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
      text += f'(estimated time: {avg_time}sec.)'
    else:
      text += f'(position in the queue: {self.queue_n}; estimated time: {self.queue_n*avg_time}sec.)'

    try:
      if not self.message:
        self.message = await self.event.reply(text)

        return

      await self.message.edit(text)
    except:
      pass

class QueueProcessor:
  def __init__(self):
    self.queue = []
    self.qsize = 0
    self.times_history = []

  async def add_task(self, event):
    self.qsize += 1

    task = Task(
      event=event,
      queue_n=self.qsize
    )

    self.queue.append(task)

    await task.update_message()

  async def loop(self):
    while True:
      while not self.queue:
        await asyncio.sleep(0)

      task = self.queue.pop(0)
      task.start_ts = time.time()

      try:
        async with AsyncTempFile('.png') as f_in:
          async with AsyncTempFile('.png') as f_out:
            await bot.download_media(task.event.photo, file=f_in.name)

            proc = await asyncio.create_subprocess_shell(
              f'python pixelization.py {f_in.name} {f_out.name} {task.pixel_size}'
            )
            await proc.communicate()

            await task.event.reply(file=f_out.name, force_document=True)
      except:
        task.error = True
      finally:
        self.times_history.append(time.time() - task.start_ts)
        if len(self.times_history) > 10:
          self.times_history.pop(0)

        self.qsize -= 1

        await task.update_message(-1)

        for task in self.queue:
          await task.update_message(-1)

bot = TelegramClient(
  'pixelization',
  config.API_ID,
  config.API_HASH
).start(bot_token=config.API_TOKEN)

processor = QueueProcessor()

@bot.on(NewMessage)
async def on_message(event):
  if not event.photo:
    await event.reply('Please, provide an image to pixelate.')

    return

  await processor.add_task(event)

with bot:
  bot.loop.create_task(processor.loop())
  bot.start()
  bot.run_until_disconnected()
