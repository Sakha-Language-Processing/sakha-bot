from os import getenv
from tempfile import NamedTemporaryFile

import torch
from librosa import load
from telebot import TeleBot
from telebot.types import Message, Update
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

repo_name = "volodya-leveryev/mms-300m-sah"
revision = "9b2efc815c0f3085e2902d8647c4c35d122e8b82"
model = Wav2Vec2ForCTC.from_pretrained(repo_name, revision=revision)
processor = Wav2Vec2Processor.from_pretrained(repo_name, revision=revision)

token = getenv("TOKEN")
bot = TeleBot(token)


@bot.message_handler(commands=["start", "help"])
def send_welcome(message: Message):
    text = "👋🏻у — киһи сахалыы саҥатын тиэкискэ кубулутар тэрил (bot, робот). Микрофоҥҥа саҥарыаххын эбэтэр аудио ыытыаххын сөп."
    bot.reply_to(message, text)


@bot.message_handler(content_types=["audio", "voice"])
def echo_all(msg: Message):
    # Скачивание записи
    record = msg.audio or msg.voice
    file = bot.get_file(record.file_id)
    content = bot.download_file(file.file_path)
    with NamedTemporaryFile() as new_file:
        new_file.write(content)
        audio, _ = load(new_file.name, sr=16_000)

    # Распознавание речи
    input_dict = processor(audio, return_tensors="pt", padding=True)
    output = model(input_dict.input_values).logits
    predictions = torch.argmax(output, dim=-1)[0]

    # Ответ бота
    bot.reply_to(msg, processor.decode(predictions))


bot.infinity_polling()
