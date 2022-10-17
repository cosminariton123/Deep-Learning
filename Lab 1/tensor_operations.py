import tensorflow as tf

def functional_aproach():
    input = tf.constant([1, 4, 2, 43, 32, 3, 5, 6])
    x = tf.reshape(input, [-1, 2])
    a, b = tf.unstack(x, num = 2, axis = 1)
    print( tf.reduce_sum(a - b) )


def wrapper_aproach():
    @tf.function
    def f(input):
        sum = 0
        for i in range(len(input)):
            if i % 2 == 0:
                sum += input[i]
            else:
                sum -= input[i]
        return sum

    input = tf.constant([1, 4, 2, 43, 32, 3, 5, 6])
    print(f(input))


def custom_model_aproach():
    input = tf.constant([1, 4, 2, 43, 32, 3, 5, 6])
    class CustomModel(tf.keras.models.Model):
        def call(self, input):
            x = tf.reshape(input, [-1, 2])
            a, b = tf.unstack(x, num = 2, axis = 1)
            return tf.reduce_sum(a - b)
    model = CustomModel()
    print(model(input))


def main():
    functional_aproach()
    wrapper_aproach()
    custom_model_aproach()

if __name__ == "__main__":
    main()