#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Get some audio, recognize speech and print the result, continuously"""

import argparse
import asyncio
import concurrent.futures

from uttermon.speech_decoder import SpeechDecoder
from uttermon.audio_input import AudioInput

model_executor = concurrent.futures.ProcessPoolExecutor(1)

decoder = SpeechDecoder()
audio_input = AudioInput()

def load_model(args):
    """Load model"""
    decoder.init(args.model)

def recognize_speech(audio):
    """Recognize speech from audio buffer"""
    return decoder.process_audio(audio)

async def process_audio():
    """Process audio stream"""
    print('Starting audio recording...')
    loop = asyncio.get_running_loop()
    async for audio_frame in audio_input.record_audio_blocks():
        lang, result, time_to_process = await loop.run_in_executor(
            model_executor,
            recognize_speech,
            audio_frame)
        if result:
            print(f'{lang}: {result} - (took {time_to_process}s)')

async def main():
    """Main function"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="the size of the whisper model to use",
        choices=["tiny", "base", "small", "medium", "large"],
        default="small"
    )

    args = parser.parse_args()

    loop = asyncio.get_running_loop()

    await loop.run_in_executor(model_executor, load_model, args)
    await loop.create_task(process_audio())

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
