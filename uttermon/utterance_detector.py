#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Detect utterances in audio stream"""

import numpy as np
import webrtcvad

class UtteranceDetector:
    """
    A class used to process audio and detect speech with a naive debounce
    mechanism to filter a moderate amount of trailing silence.

    The buffer must be a 16-bit mono PCM numpy array, and at least 10 ms long.
    The sample rate must be 8000, 16000, 32000 or 48000 Hz.
    The aggressiveness must be between 0 and 3.

    Call process_audio() in the input callback with audio buffer to detect speech.
    Stores a boolean in self.is_speaking to indicate if speech was detected.
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, window_size, samplerate, aggressiveness = 2, max_silence = 1.0):
        """Initialize voice activity detector"""
        # the window size must be 10, 20 or 30 ms
        if window_size not in [10, 20, 30]:
            raise ValueError("Window size must be 10, 20 or 30 ms")
        self.window_size = window_size

        # precalculate the window size in frames
        self.window_size_frames = int(samplerate * window_size / 1000)

        # the sample rate must be 8000, 16000, 32000 or 48000 Hz
        if samplerate not in [8000, 16000, 32000, 48000]:
            raise ValueError("Sample rate must be 8000, 16000, 32000 or 48000 Hz")
        self.samplerate = samplerate

        # the actual webrtc voice activity detector with the given aggressiveness
        if aggressiveness < 0 or aggressiveness > 3:
            raise ValueError("Aggressiveness must be between 0 and 3")
        self.vad = webrtcvad.Vad(aggressiveness)

        # the maximum silence time in seconds
        self.max_silence = max_silence
        # a boolean to keep track of the current state
        self.is_speaking = False
        # keep track of last time speech was not detected
        self.last_silence = 0

        # the buffer to store the audio data if the previous frame was
        # not enough to fill the window size
        self.buf = np.array([], dtype=np.int16)

    def process_audio(self, buf):
        """ Process a buffer and detect speech if the buffer is long enough """
        #print(f"Processing audio block of {len(buf)} frames")

        # convert the buffer to 16-bit mono PCM
        if buf.ndim == 2:
            buf = buf.mean(axis=1)
        if buf.dtype == np.float32:
            buf = (buf * 32767).astype(np.int16)

        # add the new buffer to the previous one
        self.buf = np.concatenate((self.buf[-self.window_size_frames:], buf))

        # if the buffer is not enough to process, store it for the next frame
        if len(self.buf) < self.window_size_frames:
            #print(f"Buffer is not enough to process ({len(self.buf)} frames)")
            return

        # dequeue the buffer to process it
        while len(self.buf) >= self.window_size_frames:
            buf = self.buf[:self.window_size_frames]
            self.buf = self.buf[self.window_size_frames:]

            # check if the buffer contains speech
            #print(f"Detecting speech in {len(buf)} frame window")
            is_speaking = self.vad.is_speech(buf.tobytes(), self.samplerate)

            # if speech was detected, reset the silence counter
            if is_speaking:
                self.last_silence = 0
            else:
                self.last_silence += self.window_size / 1000

            # check if the silence time is too long
            if self.last_silence > self.max_silence:
                is_speaking = False

            # adjust the is_speaking state
            if is_speaking != self.is_speaking:
                self.is_speaking = is_speaking

    def reset(self):
        """Reset the detector"""
        self.is_speaking = False
        self.last_silence = 0
        self.buf = np.array([], dtype=np.int16)
