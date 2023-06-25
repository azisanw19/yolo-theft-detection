"""
Generate images to get person Bounding in images

__author__      = "Aziz Anwar"
__copyright__   = "Copyright 2023, Canwar"
"""

import os
import cv2


def initialize_haarcascade_full_body_detection():
    """
    initialize mobile net to automatic person segmentation

    :return: model
    """
    full_body = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

    return full_body


def bounding_person(model, img):
    """
    Bounding person

    :param model: haarcascade full body
    :param img: image
    :return: image with person bounding
    """
    bodies = model.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

    return bodies


def main():
    model = initialize_haarcascade_full_body_detection()

    path = os.path.join("dataset")

    is_exist_folder_segmentation = os.path.exists(os.path.join(path, "image-bounding-haarcascade"))
    if not is_exist_folder_segmentation:
        os.makedirs("dataset/image-bounding-haarcascade/theft")
        os.makedirs("dataset/image-bounding-haarcascade/guest")

    for folder in os.listdir(os.path.join(path, "original")):
        image_path = os.path.join(path, "original")
        image_path = os.path.join(image_path, folder)

        save_path = os.path.join(path, "image-bounding-haarcascade")
        save_path = os.path.join(save_path, folder)

        for image in os.listdir(image_path):
            img = cv2.imread(os.path.join(image_path, image))

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            bounding = bounding_person(model, gray_img)

            count = 0
            for x, y, w, h in bounding:
                save_path_image = os.path.join(save_path, "{}{}".format(count, image))

                img_person = img[y:y + h, x:x + w]
                cv2.imwrite(save_path_image, img_person)
                count += 1
                print("[INFO]: Success saving image to {}".format(save_path_image))


if __name__ == '__main__':
    main()
