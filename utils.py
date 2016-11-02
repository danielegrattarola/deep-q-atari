import numpy as np
from PIL import Image

IMG_SIZE = None


# Functions
def preprocess_observation(obs):
    global IMG_SIZE

    image = Image.fromarray(obs, 'RGB').convert('L').resize(IMG_SIZE)  # Convert to gray-scale and resize according to PIL coordinates
    return np.asarray(image.getdata(), dtype=np.uint8).reshape(image.size[1],
                                                               image.size[0])  # Convert to array and return


def get_next_state(current, obs):
    # Next state is composed by the last 3 images of the previous state and the new observation
    return np.append(current[1:], [obs], axis=0)
