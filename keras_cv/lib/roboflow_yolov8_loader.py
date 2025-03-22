import os
import typing

import tensorflow as tf
import tqdm


class RoboflowYOLOv8Loader:
    """RoboflowYOLOv8Loader is a custom data loader that can be used to load
    datasets that are exported from Roboflow in YOLO v8 format.
    """

    class_mapping: dict
    data_dir: str
    batch_size: int = 8

    def __init__(self, class_list: list, data_dir: str, batch_size: int = 8) -> None:
        """constructor for the custom data loader.
        Args:
            class_mapping (dict): classes and numerical representation.
            data_dir (str): top level data directory assuming structure below.
                ├── test
                │   ├── images
                │   └── labels
                ├── train
                │   ├── images
                │   └── labels
                └── valid
                    ├── images
                    └── labels
        """
        assert len(class_list) > 0, "Class list must be greater than 0."

        self.class_mapping = dict(zip(range(len(class_list)), class_list))
        self.data_dir = data_dir
        self.batch_size = batch_size

    def load_dataset(
        self, name: str | None = None
    ) -> tf.data.Dataset | typing.Tuple[tf.data.Dataset]:
        """load_dataset will load a full dataset or a specific split if desired.

        Args:
            name (str | None, optional): split name (train, valid, test).
                    Defaults to None, which will load all 3.

        Returns:
            tf.data.Dataset | typing.Tuple[tf.data.Dataset]: one or a tuple
                of datasets in the following order (train, valid, test).
        """
        if name:
            return self.__load_dataset(name)
        else:
            # load all 3
            train = self.__load_dataset("train", shuffle=True)
            val = self.__load_dataset("valid")
            test = self.__load_dataset("test")
            return (train, val, test)

    def __load_dataset(self, name: str, shuffle: bool = False) -> tf.data.Dataset:
        """__load_dataset is a private function
        that will load one specific split of a dataset
        given the name. (train, valid, test).

        Args:
            name (str): dataset split name (train, valid, test)

        Returns:
            tf.data.Dataset: the tensorflow dataset.
        """
        # validate inputs
        VALID_NAMES = ["train", "valid", "test"]
        assert name.lower() in VALID_NAMES, "name must be one of: " + str(VALID_NAMES)

        # define image and label paths
        split_path = f"{self.data_dir}/{name.lower()}"
        image_dir = f"{split_path}/images"
        label_path = f"{split_path}/labels"
        image_paths = sorted(
            [
                os.path.join(image_dir, f)
                for f in os.listdir(image_dir)
                if f.endswith(".jpg")
            ]
        )
        label_paths = sorted(
            [
                os.path.join(label_path, f)
                for f in os.listdir(label_path)
                if f.endswith(".txt")
            ]
        )

        # load images and data using yolo v8 labels.
        bbox = []
        classes = []
        for label in tqdm.tqdm(label_paths):
            boxes, class_ids = RoboflowYOLOv8Loader.parse_yolo_label(label)
            bbox.append(boxes)
            classes.append(class_ids)

        assert (
            len(image_paths) == len(bbox) == len(classes)
        ), "inputs must be the same length on the first dimension."

        # can contain lists of varying lengths, so its recommended to use ragged tensors.
        bbox = tf.ragged.constant(bbox)
        classes = tf.ragged.constant(classes)
        image_paths = tf.ragged.constant(image_paths)

        # load into tf dataset and process into standard structure.
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, classes, bbox))
        dataset = dataset.map(
            RoboflowYOLOv8Loader.to_dataset_dict, num_parallel_calls=tf.data.AUTOTUNE
        )
        return dataset

    @staticmethod
    def load_image(image_path: str) -> tf.Tensor:
        """the image in tensor format"""
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize_with_pad(image, 640, 640)
        image = tf.cast(image, dtype=tf.float32) / 255.0  # Normalize to [0, 1]
        return image

    @staticmethod
    def to_dataset_dict(
        image_path: str, classes: tf.RaggedTensor, bbox: tf.RaggedTensor
    ) -> dict:
        """data loaded in standard pipeline format:
        {
            "images": Tensor(),
            "bounding_boxes": {
                "classes": Tensor(),
                "boxes": RaggedTensor()
            }
        }
        Reference: https://keras.io/examples/vision/yolov8/

        Args:
            image_path (str): path to image
            classes (tf.RaggedTensor): classes in the image
            bbox (tf.RaggedTensor): bounding boxes in the image

        Returns:
            dict: standard tf pipeline format
        """
        image = RoboflowYOLOv8Loader.load_image(image_path)
        bounding_boxes = {
            "classes": tf.cast(classes, dtype=tf.float32),
            "boxes": bbox,
        }
        return {"images": image, "bounding_boxes": bounding_boxes}

    @staticmethod
    def parse_yolo_label(
        label_path: str, image_width: int = 640, image_height: int = 640
    ) -> typing.Tuple[list]:
        """parse_yolo_label will read an input txt file in the yolov8 format
        (center_xywh) and return it in xmin, ymin, xmax, ymax (Pascal VOC format).

        Args:
            label_path (str): path to label file.
            image_width (int, optional). Defaults to 640.
            image_height (int, optional). Defaults to 640.

        Returns:
            typing.Tuple[list]: boxes, class_ids
        """
        with open(label_path, "r") as f:
            lines = f.readlines()

        boxes = []
        class_ids = []

        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])

            # YOLOv8 bounding box in [x_center, y_center, width, height]
            x_center, y_center, width, height = parts[1:]

            # Convert center_xywh to xmin, ymin, xmax, ymax (Pascal VOC format)
            xmin = (float(x_center) - float(width) / 2) * image_width
            ymin = (float(y_center) - float(height) / 2) * image_height
            xmax = (float(x_center) + float(width) / 2) * image_width
            ymax = (float(y_center) + float(height) / 2) * image_height

            boxes.append([xmin, ymin, xmax, ymax])
            class_ids.append(class_id)

        return boxes, class_ids
