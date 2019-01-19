from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as resnet_preprocess
from keras.models import Model
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os


IMAGE_DIR = './data/book_cover_images'
FEATURE_OUT_DIR = './features'


def extract_vgg_features(feature_dump='vgg_book_cover_features_2.csv'):
    model = VGG16(weights='imagenet', include_top=False)
    features = {}
    for img_file in os.listdir(IMAGE_DIR):
        if img_file.split('.')[-1] in ['jpg', 'jpeg', 'png','JPG']:
            img_path = os.path.join(IMAGE_DIR, img_file)
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            feature_vector = model.predict(x).flatten()
            # print (feature_vector.shape)
            features[img_file] = feature_vector.tolist()
        else:
            print('Escaped file {}'.format(img_file))

    with open(os.path.join(FEATURE_OUT_DIR, feature_dump), 'w') as f_out:
        for x, feature_vector in features.items():
            f_out.write(",".join([x] + list(map(str, feature_vector))) + '\n')

    print("Done dumping the feature vectors")


def extract_resnet_features(feature_dump='resnet_book_cover_features_2.csv'):
    model = ResNet50(weights='imagenet', include_top=False)
    features = {}

    for img_file in os.listdir(IMAGE_DIR):
        if img_file.split('.')[-1] in ['jpg', 'jpeg', 'png','JPG']:

            img_path = os.path.join(IMAGE_DIR, img_file)
            img = image.load_img(img_path, target_size=(224, 224))

            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = resnet_preprocess(x)
            feature_vector = model.predict(x).flatten()
            features[img_file] = feature_vector.tolist()
        else:
            print('Escaped file {}'.format(img_file))

    with open(os.path.join(FEATURE_OUT_DIR, feature_dump), 'w') as f_out:
        for x, feature_vector in features.items():
            f_out.write(",".join([x] + list(map(str, feature_vector))) + '\n')

    print("Done dumping the feature vectors")


if __name__ == '__main__':
    extract_resnet_features()
    extract_vgg_features()
