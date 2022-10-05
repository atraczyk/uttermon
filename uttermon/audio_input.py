#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Open a stream and record audio blocks if speech is detected"""

import asyncio
import sounddevice as sd
import numpy as np

from .utterance_detector import UtteranceDetector

def trapezoidal_window(length, taper=0.1):
    """Trapezoidal window function as Numpy array"""
    window = np.ones(length)
    taper_length = int(length * taper)
    window[:taper_length] = np.linspace(0, 1, taper_length)
    window[-taper_length:] = np.linspace(1, 0, taper_length)
    return window

class AudioInput:
    """Audio input"""

    def __init__(self, samplerate = 16000):
        self.samplerate = samplerate

    async def inputstream_generator(self, **kwargs):
        """
        Generator that yields blocks of input data as NumPy arrays.

        This is a modified version of the inputstream_generator from the
        sounddevice module.
        """
        q_in = asyncio.Queue()
        loop = asyncio.get_event_loop()

        # pylint: disable=unused-argument
        def callback(indata, frame_count, time_info, status):
            loop.call_soon_threadsafe(q_in.put_nowait, (indata.copy(), status))

        stream = sd.InputStream(
            callback=callback,
            channels=1,
            samplerate=self.samplerate,
            blocksize=119,
            **kwargs)
        with stream:
            while True:
                indata, status = await q_in.get()
                yield indata, status


    async def record_audio_blocks(self, **kwargs):
        """ Record audio blocks from input stream """
        vad = UtteranceDetector(20, self.samplerate, 3, 1.0)

        # calculate a minumum block size to yield
        min_audio_buffer_len = int(self.samplerate * 0.5 * vad.max_silence)

        # initialize the audio buffer
        audio_buf = np.ndarray(1, dtype=np.float32)
        async for indata, status in self.inputstream_generator(**kwargs):
            if status:
                print(status)

            # if speech is detected, append the data to the buffer
            vad.process_audio(indata)
            if vad.is_speaking:
                audio_buf = np.append(audio_buf, indata)
            # if not speaking anymore and we have some audio, yield it
            elif not vad.is_speaking and len(audio_buf) > min_audio_buffer_len:
                # copy the audio buffer and apply a trapezoidal window function
                # to avoid discontinuities. it seems to prevent the decoder from
                # returning garbage results.
                window = trapezoidal_window(len(audio_buf)).astype(np.float32)
                copy = audio_buf * window

                # reset audio buffer to empty
                audio_buf = np.ndarray(1, dtype=np.float32)
                yield copy
