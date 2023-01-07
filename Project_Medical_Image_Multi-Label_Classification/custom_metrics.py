import tensorflow as tf

def convert_from_matrix_to_vector_of_appearence(matrix):
        return tf.reduce_sum(tf.unstack(matrix, axis=1), axis=1)

def compute_tp_fp_tp(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_pred = y_pred * 2
    y_pred = tf.floor(y_pred)

    tp_matrix = y_pred * y_true

    fp_matrix = y_pred - tp_matrix

    fn_matrix = (y_pred + y_true) - 2 * tp_matrix - fp_matrix

    tp = convert_from_matrix_to_vector_of_appearence(tp_matrix)
    fp = convert_from_matrix_to_vector_of_appearence(fp_matrix)
    fn = convert_from_matrix_to_vector_of_appearence(fn_matrix)

    return tp, fp, fn

def mean_f1_score(y_true, y_pred):

    tp, fp, fn = compute_tp_fp_tp(y_true, y_pred)

    f1 = 2 * tp / (2 * tp + fp + fn)
    f1_mean = tf.reduce_mean(f1)

    return f1_mean

def first_class_precision(y_true, y_pred):
    tp, fp, fn = compute_tp_fp_tp(y_true, y_pred)

    return (tp / (tp + fp))[0]

def second_class_precision(y_true, y_pred):
    tp, fp, fn = compute_tp_fp_tp(y_true, y_pred)

    return (tp / (tp + fp))[1]

def third_class_precision(y_true, y_pred):
    tp, fp, fn = compute_tp_fp_tp(y_true, y_pred)

    return (tp / (tp + fp))[2]

def average_class_precision(y_true, y_pred):
    tp, fn, fp = compute_tp_fp_tp(y_true, y_pred)

    return tf.reduce_mean(tp / (tp + fp))

CUSTOM_METRICS = [mean_f1_score, first_class_precision, second_class_precision, third_class_precision, average_class_precision]
