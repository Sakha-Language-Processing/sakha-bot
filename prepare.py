from transformers import pipeline

pipe = pipeline(
    task="automatic-speech-recognition",
    model="volodya-leveryev/mms-300m-sah",
)
