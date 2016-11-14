from vgg19 import VGG19
from keras.preprocessing import image
from imagenet_utils import preprocess_input
import numpy as np

class VGG19FE:

    def __init__(self):
        self.model = VGG19(weights='imagenet', include_top=False)

    def get_features(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return self.model.predict(x)