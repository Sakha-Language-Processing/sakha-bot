from os import getenv
from tempfile import NamedTemporaryFile

import soundfile
import torch
from librosa import load
from telebot import TeleBot
from telebot.types import Message
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from sys import path
path.append("vits")
import commons
import utils
from models import SynthesizerTrn

repo_name = "volodya-leveryev/mms-300m-sah"
revision = "9b2efc815c0f3085e2902d8647c4c35d122e8b82"
model = Wav2Vec2ForCTC.from_pretrained(repo_name, revision=revision)
processor = Wav2Vec2Processor.from_pretrained(repo_name, revision=revision)

vocab_file = "./sah/vocab.txt"
config_file = "./sah/config.json"
hps = utils.get_hparams_from_file(config_file)
with open(vocab_file, encoding="utf-8") as f:
    data = (x.replace("\n", "") for x in f.readlines())
    symbols_to_id = {s: i for i, s in enumerate(data)}

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

net_g = SynthesizerTrn(
    len(symbols_to_id),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model,
)
net_g.to(device)
_ = net_g.eval()
g_pth = f"./sah/G_100000.pth"
_ = utils.load_checkpoint(g_pth, net_g, None)

token = getenv("TG_TOKEN", "")
bot = TeleBot(token)


def filter_oov(text):
    return "".join(list(filter(lambda x: x in symbols_to_id, text)))


def text_to_sequence(text):
    sequence = []
    clean_text = text.strip()
    for symbol in clean_text:
        symbol_id = symbols_to_id[symbol]
        sequence += [symbol_id]
    return sequence


@bot.message_handler(commands=["start", "help"])
def send_welcome(msg: Message):
    text = "👋🏻 Бу — киһи сахалыы саҥатын тиэкискэ кубулутар тэрил (bot, робот). Микрофоҥҥа саҥарыаххын эбэтэр аудио ыытыаххын сөп."
    bot.reply_to(msg, text)


@bot.message_handler(content_types=["audio", "voice", "text"])
def speech_to_text(msg: Message):
    if msg.content_type in ("audio", "voice"):
        # Скачивание записи
        record = msg.audio or msg.voice
        file = bot.get_file(record.file_id)
        content = bot.download_file(file.file_path)
        with NamedTemporaryFile() as tmp_file:
            tmp_file.write(content)
            audio, _ = load(tmp_file.name, sr=16_000)

        # Распознавание речи
        input_dict = processor(audio, return_tensors="pt", padding=True)
        output = model(input_dict.input_values).logits
        predictions = torch.argmax(output, dim=-1)[0]

        # Ответ бота
        bot.reply_to(msg, processor.decode(predictions))

    elif msg.content_type in ("text",):
        text = filter_oov(msg.text.lower())
        text_norm = text_to_sequence(text)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)

        with torch.no_grad():
            x_tst = text_norm.unsqueeze(0).to(device)
            x_tst_lengths = torch.LongTensor([text_norm.size(0)]).to(device)
            hyp = net_g.infer(
                x_tst, x_tst_lengths, noise_scale=.667,
                noise_scale_w=0.8, length_scale=1.0,
            )[0][0,0].cpu().float().numpy()

        with NamedTemporaryFile() as tmp_file:
            soundfile.write(tmp_file.name, hyp, 16_000, format="WAV")
            bot.send_audio(msg.chat.id, tmp_file, reply_to_message_id=msg.id)


bot.infinity_polling()
