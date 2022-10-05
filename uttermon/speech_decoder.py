#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Class for speech recognition using OpenAI's whisper API"""

import time
import whisper

class SpeechDecoder:
    """Speech recognizer using OpenAI's whisper API"""

    def __init__(self, task = "translate"):
        self.audio_options = whisper.DecodingOptions(task = task)
        self.model = None


    def init(self, model_size = "small"):
        """Initialize the speech decoder"""
        print(f'Loading the {model_size} whisper model...')
        self.model = whisper.load_model(model_size)


    def process_audio(self, audio):
        """Recognize speech from audio buffer"""
        start_time = time.time()
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        result = whisper.decode(self.model, mel, self.audio_options)
        time_to_process = time.time() - start_time
        if (result.no_speech_prob < .5
            and not result.text.startswith('Thank you for watching')
            and not result.text.startswith('Thanks for watching')):
            return result.language, result.text, time_to_process
        return None, '', None
