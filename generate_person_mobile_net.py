"""
Generate images to get person Bounding in images

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


def bounding_person(model, img):
    """
    Bounding person

    :param model: model mobile net
    :param img: image
    :return: image with person bounding
    """
    class_index, confidence, box = model.detect(img)
    boundary = []

    for index, confi, boxes in zip(class_index, confidence, box):
        if index == 1:
            boundary.append(boxes)
    return boundary


def main():
    model = initialize_mobile_net_detection()

    path = os.path.join("dataset")

    is_exist_folder_segmentation = os.path.exists(os.path.join(path, "image-bounding"))
    if not is_exist_folder_segmentation:
        os.makedirs("dataset/image-bounding/theft")
        os.makedirs("dataset/image-bounding/guest")

    for folder in os.listdir(os.path.join(path, "original")):
        image_path = os.path.join(path, "original")
        image_path = os.path.join(image_path, folder)

        save_path = os.path.join(path, "image-bounding")
        save_path = os.path.join(save_path, folder)

        for image in os.listdir(image_path):
            img = cv2.imread(os.path.join(image_path, image))
            bounding = bounding_person(model, img)

            count = 0
            for x, y, w, h in bounding:
                save_path_image = os.path.join(save_path, "{}{}".format(count, image))

                img_person = img[y:y + h, x:x + w]
                cv2.imwrite(save_path_image, img_person)
                count += 1
                print("[INFO]: Success saving image to {}".format(save_path_image))


if __name__ == '__main__':
    main()
