import tensorflow as tf

def preprocess_image(image, label):
    image = tf.cast(image, tf.float32)
    image = image / 255.

    return image, label
