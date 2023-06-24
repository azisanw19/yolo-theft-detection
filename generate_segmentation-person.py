"""
Generate images to get person segmentation in images

__author__      = "Aziz Anwar"
__copyright__   = "Copyright 2023, Canwar"
"""

import os
import cv2


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

    for index, confi, boxes in zip(class_index, confidence, box):
        if index == 1:
            cv2.rectangle(img, boxes, color=(0, 255, 0), thickness=2)
    return img


def main():
    model = initialize_mobile_net_detection()

    path = os.path.join("dataset")

    is_exist_folder_segmentation = os.path.exists(os.path.join(path, "segmentation"))
    if not is_exist_folder_segmentation:
        os.makedirs("dataset/segmentation/theft")
        os.makedirs("dataset/segmentation/guest")

    print(path)

    for folder in os.listdir(os.path.join(path, "original")):
        image_path = os.path.join(path, "original")
        image_path = os.path.join(image_path, folder)

        save_path = os.path.join(path, "segmentation")
        save_path = os.path.join(save_path, folder)

        for image in os.listdir(image_path):
            img = cv2.imread(os.path.join(image_path, image))
            img = segmentation_person(model, img)
            cv2.imwrite(os.path.join(save_path, image), img)
            print(f"[INFO]\r: Success saving image to {save_path.join(image)}")

    print(path)


if __name__ == '__main__':
    main()
