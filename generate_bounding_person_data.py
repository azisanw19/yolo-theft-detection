"""
Generate images to get person segmentation in images

__author__      = "Aziz Anwar"
__copyright__   = "Copyright 2023, Canwar"
"""

import os
import cv2
import pandas as pd


def initialize_mobile_net_detection():
    """
    initialize mobile net to automatic person segmentation

    :return: model
    """
    config_file = f"model/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    frozen_model = f"model/frozen_inference_graph.pb"

    model = cv2.dnn_DetectionModel(frozen_model, config_file)
    model.setInputSize(320, 320)
    model.setInputScale(1.0 / 127.5)
    model.setInputMean((127.5, 127.5, 127.5))
    model.setInputSwapRB(True)

    return model


def segmentation_person(model, img):
    """
    Segmentation person

    :param model: model mobile net
    :param img: image
    :return: image with person segmentation
    """
    class_index, confidence, box = model.detect(img)
    boundary = None, None, None, None

    for index, confi, boxes in zip(class_index, confidence, box):
        if index == 1:
            boundary = boxes
    return boundary


def main():
    """

    :param x: x axis of image
    :param y: y axis of image
    :param w: weight (the width of the person from the x axis)
    :param h: height (the height of the person from the y axis)

    :return:
    """
    model = initialize_mobile_net_detection()

    path = os.path.join("dataset")

    is_exist_folder_bounding = os.path.exists(os.path.join(path, "bounding"))
    if not is_exist_folder_bounding:
        os.makedirs("dataset/bounding")


    path_image = []
    x = []
    y = []
    w = []
    h = []
    target_label = []

    for folder in os.listdir(os.path.join(path, "original")):
        image_path = os.path.join(path, "original")
        image_path = os.path.join(image_path, folder)

        for image in os.listdir(image_path):
            image_path_file = os.path.join(image_path, image)
            img = cv2.imread(image_path_file)

            (temp_x, temp_y, temp_w, temp_h) = segmentation_person(model, img)

            if temp_x is not None:
                path_image.append(image_path_file)
                x.append(temp_h)
                y.append(temp_y)
                w.append(temp_w)
                h.append(temp_h)
                target_label.append(folder)
                print("[INFO]: Success adding data image to {}".format(image_path_file))
            else:
                print("[INFO]: Failed adding data image to {}".format(image_path_file))
    data_image = pd.DataFrame({
        "path": path_image,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "label": target_label
    })

    save_path = os.path.join(path, "bounding")
    data_image.to_csv(f"{save_path}/dataset.csv", index=False)
    print("[INFO]: Saving CSV to {}/dataset.csv".format(save_path))


if __name__ == '__main__':
    main()
