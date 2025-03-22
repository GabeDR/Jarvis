import speech_recognition as sr

class AudioToTextRecorder:
    def __init__(self, spinner=False, model="tiny.en", language="en", post_speech_silence_duration=0.1, silero_sensitivity=0.4):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.spinner = spinner
        self.model = model
        self.language = language
        self.post_speech_silence_duration = post_speech_silence_duration
        self.silero_sensitivity = silero_sensitivity

    def text(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            print("Listening...")
            audio = self.recognizer.listen(source)
        try:
            print("Recognizing...")
            text = self.recognizer.recognize_google(audio, language=self.language)
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Could not request results; {e}"
