# Visible Watermark Information Classification (VWIC)

Visible watermarking is a technique typically used by content owners or distributors that are legally contracted to distribute or sell content on behalf of the owners.  The technique involves adding information to the content using semi-transparent text and/or logos. Watermarks are often used to establish ownership and discourage unauthorized use of the content thus protecting the intellectual property. When watermarking images, the owners or distributors often include different types of information such as a logo, a unique identifier, contact information, notices and more. The existing research [1], [2] and datasets available for the visible watermarks domain focus on detecting watermarks for the purposes of removing them.   Extending current research, this project explores the application of advanced object detection techniques to accurately identify and categorize information embedded in visible watermarks. A key contribution is the development of the Visible Watermark Information Classification (VWIC) dataset (the gather your own dataset project template), a collection of images showcasing a range of watermarks and associated information. The VWIC dataset served as the foundation for training and evaluating state-of-the-art models, including YOLOv8 and RetinaNet, with mean Average Precision (mAP) as the primary metric. The results demonstrate the successful application of these models, achieving a 68.9% mAP (IoU=50:90) / 96.9% mAP (IoU=50) in detecting and classifying watermark information using the new VWIC dataset.

The dataset contains 4 distinct classes of visible watermarks (Logo, Identifier, Contact, and Notice) and is used to fine-tune pre-trained models using You Only Look Once (YOLO) and RetinaNet object detection algorithms.


Keywords: visible watermarking; watermark detection; content protection; object detection; classification; computer vision; machine learning; YOLO; RetinaNet

[1]	Cheng, D., Li, X., Li, W., Lu, C., Li, F., Zhao, H. & Zheng, W. 2018, "Large-Scale Visible Watermark Detection and Removal with Deep Convolutional Networks" in Pattern Recognition and Computer Vision Springer International Publishing AG, , pp. 27–40.

[2]	Santoyo-Garcia, H., Fragoso-Navarro, E., Reyes-Reyes, R., Sanchez-Perez, G., Nakano-Miyatake, M. & Perez-Meana, H. 04/2017, "An automatic visible watermark detection method using total variation", IEEE, pp. 1.

The dataset: https://universe.roboflow.com/vwic/visible-watermark-information-classification-voc2007

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
