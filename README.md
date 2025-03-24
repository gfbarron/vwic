# Visible Watermark Information Classification (VWIC)

This project contributes a new dataset using images from Pascal VOC 2007 and applying multiple visible watermarks.

The dataset contains 4 distinct classes of visible watermarks (Logo, Identifier, Contact, and Notice) and is used
to fine-tune pre-trained models using You Only Look Once (YOLO) and RetinaNet object detection algorithms.

## Acknowledgements

1. Pascal VOC 2007.
2. Ultralytics.
3. Keras CV, Keras and TensorFlow.
4. Roboflow.com
5. MakeWatermark.com

## Dependencies

* Python: `3.12`

## Getting Started

The required dependencies are available in `requirements.txt` and have been tested on Windows 10 using WSL2 (Ubuntu).

Each implementation has it's own requirements file, so be sure to use the correct one.

* `python -m venv .venv`
* `source .venv/bin/activate`
* `pip install --upgrade pip`
* `pip install -r requirements.txt`

Example: 

```bash
cd ultralytics
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
