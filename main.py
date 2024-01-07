import tkinter as tk
from tkinter import Label, Button
import speech_recognition as sr
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download the VADER lexicon
nltk.download('vader_lexicon')

class VoiceSentimentAnalyzerApp:
    def __init__(self, master):
        self.master = master
        master.title("Voice Sentiment Analyzer")

        self.sentiment_label = Label(master, text="Sentiment:")
        self.sentiment_label.pack()

        self.listen_button = Button(master, text="Start Listening", command=self.get_voice_tone)
        self.listen_button.pack()

    def get_text_sentiment(self, text):
        sia = SentimentIntensityAnalyzer()
        sentiment_score = sia.polarity_scores(text)['compound']

        if sentiment_score >= 0.05:
            return "positive"
        elif sentiment_score <= -0.05:
            return "negative"
        else:
            return "neutral"

    def get_voice_tone(self):
        recognizer = sr.Recognizer()

        with sr.Microphone() as source:
            print("Please say something...")
            recognizer.adjust_for_ambient_noise(source)
            audio_data = recognizer.listen(source, timeout=5)

        try:
            text = recognizer.recognize_google(audio_data)
            print(f"Text from voice: {text}")
            sentiment = self.get_text_sentiment(text)
            self.sentiment_label.config(text=f"Sentiment: {sentiment}")
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceSentimentAnalyzerApp(root)
    root.mainloop()
