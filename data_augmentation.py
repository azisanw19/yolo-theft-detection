"""
Augmentation image to get more images for training

__author__      = "Aziz Anwar"
__copyright__   = "Copyright 2023, Canwar"
"""

import os
from keras.preprocessing.image import ImageDataGenerator


def initialize_data_generator():
    return ImageDataGenerator(
        width_shift_range=1.0,
        height_shift_range=1.0,
        brightness_range=[0.2, 0.8],
        shear_range=0.2,
        zoom_range=[0.2, 0.6],
        horizontal_flip=True,
    )


def main():
    path = os.path.join("dataset")

    is_exist_folder_augmentation = os.path.exists(os.path.join(path, "augmentation"))
    if not is_exist_folder_augmentation:
        os.makedirs("dataset/augmentation/theft")
        os.makedirs("dataset/augmentation/guest")

        print("[INFO] : Create folder augmentation/theft success")
        print("[INFO] : Create folder augmentation/guest success")

    data_generator = initialize_data_generator()

    number_of_one_sample_image = 32

    data_augmenatation = data_generator.flow_from_directory(
        "dataset/original",
        batch_size=number_of_one_sample_image,
        save_to_dir="dataset/augmentation",
        save_format='jpg'
    )

    i = 0
    for batch, img_path in (data_augmenatation, data_augmenatation.batch_size):
        print(img_path)

        i += 1
        print("[INFO]: Success saving image augmentation")
        if i >= number_of_one_sample_image:
            break


if __name__ == '__main__':
    main()
