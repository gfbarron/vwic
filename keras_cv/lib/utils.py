import keras_cv
import pandas as pd
import tqdm
import tensorflow as tf


def visualize_dataset(inputs, class_mapping, rows, cols, bounding_box_format):
    """used to visualize images in a dataset

    Reference: https://keras.io/examples/vision/yolov8/
    Args:
        inputs (Dataset): dataset
        class_mapping (dict): name to number mapping for classes
        rows (int): number of rows to display
        cols (int): number of columns to display
        bounding_box_format (str): bounding box format eg. xyxy
    """
    inputs = next(iter(inputs.take(1)))
    images, bounding_boxes = inputs[0], inputs[1]

    # Denormalize images to 0-255 since value_range parameter doesn't seem to work
    images_denormalized = tf.cast(images * 255.0, dtype=tf.uint8)

    keras_cv.visualization.plot_bounding_box_gallery(
        images_denormalized,
        value_range=(0, 255),
        rows=rows,
        cols=cols,
        y_true=bounding_boxes,
        scale = 6,
        font_scale = 0.8,
        line_thickness=2,
        dpi = 100,
        bounding_box_format=bounding_box_format,
        class_mapping=class_mapping,
        true_color = (192, 57, 43))
    

def view_splits(train, val, test):
    '''print percentage of total for all splits'''
    train_count = len(train)
    val_count = len(val)
    test_count = len(test)
    total = train_count + val_count + test_count

    def get_percentage(split_count, total, ndigits = 2):
        '''get percentage of total for a specific split'''
        pct = split_count / total * 100
        return round(pct, ndigits)
    
    counts = {
        "train": get_percentage(train_count, total),
        "val": get_percentage(val_count, total),
        "test": get_percentage(test_count, total)
    }
    
    df = pd.DataFrame(list([[split, value] for split, value in counts.items()]), columns=['Split', 'Percentage'])
    print("=================================")
    print("         Dataset Splits          ")
    print("=================================")
    print(df)


def evaluate_coco_metrics(model, test_dataset, bounding_box_format):
    """Run evaluation on the test set to collect COCO metrics.

    Args:
        model (Model): trained model
        test_dataset (Dataset): dataset to run evaluation against
        coco_metrics (BoxCOCOMetrics): metrics instance to calc results
        bounding_box_format (str): format of bounding boxes
    """
    coco_metrics = keras_cv.metrics.BoxCOCOMetrics(
        bounding_box_format=bounding_box_format,
        evaluate_freq=1e9
    )
    
    for images, y_true in tqdm.tqdm(test_dataset, desc="evaluating..."):
        y_pred = model.predict(images)
        coco_metrics.update_state(y_true, y_pred)

    results = coco_metrics.result(force=True)

    # results returns a tensor as a value so must be converted to display nicely in pandas.
    df = pd.DataFrame(list([[metric, float(value.numpy())] for metric, value in results.items()]), columns=['Metric', 'Value'])
    print("=================================")
    print("         COCO Metrics            ")
    print("=================================")
    print(df)

