"""
Yolo video streaming

__author__      = "Aziz Anwar"
__copyright__   = "Copyright 2023, Canwar"
"""

import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import datetime
import telegram
import asyncio
import config


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
            print(f"confidence {confi}")
            boundary.append(boxes)
    return boundary


def initialize_theft_model():
    """
    load model theft detection

    :return: (model, oneHotEncoding)
    """
    print("[INFO]: Load model detector ...")
    path = os.path.join("model")
    model = load_model("{}/model-classification.h5".format(path))
    encoding = pickle.loads(open("{}/one-hot-classification.pickle".format(path), "rb").read())
    print("[INFO]: Success load model detector ...")

    return model, encoding


async def send_image(bot, chat_id, img, caption):
    """
    Send image to telegram
    :param bot: Telegram bot
    :param chat_id: chat id send to telegram
    :param img: img send to telegram
    :return:
    """

    path_image_save = os.path.join("image", "{}.jpg".format(datetime.datetime.now()))
    cv2.imwrite(path_image_save, img)

    await bot.send_photo(
        chat_id=chat_id,
        photo=path_image_save,
        caption=caption
    )


def live_transmission():
    """
    live transmission
    :return:
    """
    cam = cv2.VideoCapture(config.URL)

    bot = telegram.Bot(config.TOKEN)
    chat_group_id = config.CHAT_GROUP_ID

    is_image_path_exist = os.path.exists("image")
    if not is_image_path_exist:
        os.makedirs("image")

    model = initialize_mobile_net_detection()
    model_theft, encoding = initialize_theft_model()
    count = 0
    while True:
        # Read the frame
        _, img = cam.read()

        if count % 500 == 0:
            boundary = bounding_person(model, img)

            for x, y, w, h in boundary:
                img_person = img[y:y + h, x:x + w]
                person_resize = cv2.resize(img_person, (320, 160))
                person_resize = np.expand_dims(person_resize, axis=0)

                label = model_theft.predict(person_resize)
                result = encoding.inverse_transform(label)

                if result == "theft":
                    cv2.rectangle(img, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2)
                    cv2.putText(img, "Detected {}".format(result[0][0]), (x + 5, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0),
                                thickness=2)
                    asyncio.get_event_loop().run_until_complete(
                        send_image(
                            bot,
                            chat_id=chat_group_id,
                            img=img,
                            caption="Alert! Theft detected at {}! ".format(datetime.datetime.now())
                        )
                    )
                else:
                    cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
                    cv2.putText(img, "Detected {}".format(result[0][0]), (x + 5, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0),
                                thickness=2)
                    asyncio.get_event_loop().run_until_complete(
                        send_image(
                            bot,
                            chat_id=chat_group_id,
                            img=img,
                            caption="Guest detected at {}! ".format(datetime.datetime.now())
                        )
                    )

        count += 1
        # live window
        cv2.imshow("live transmission", img)

        if count == 1000_000:
            count = 0

        # Keluar window
        key = cv2.waitKey(20)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    live_transmission()
