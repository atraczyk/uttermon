# Uttermon

Uttermon aims to provide a close-to real-time transcription of your voice to text. It is a work in progress and is not yet ready for production use.

## Installation

For now you need to install the package from source. You can do this by cloning the repository and doing the conda environment setup:

```bash
git clone https://github.com/atraczyk/uttermon.git
cd uttermon
conda env create -f environment.yml
conda activate uttermon
```

## Usage

Its not ready for easy integration yet, but you can run the demo by doing:

```bash
python main.py --model medium
```

## License

Uttermon is licensed under the MIT license. See [LICENSE](https://spdx.org/licenses/MIT.html) for more information.

## Acknowledgements

Uttermon borrows heavily from the [voice2img](https://github.com/aberaud/voice2img) project. Uttermon also uses the following open source projects:

* [python-sounddevice](https://github.com/spatialaudio/python-sounddevice)
* [whisper](https://github.com/openai/whisper)
* [webrtcvad](https://github.com/wiseman/py-webrtcvad)
* [numpy](https://github.com/numpy/numpy)