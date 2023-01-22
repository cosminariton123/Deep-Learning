import tensorflow as tf

def convert_from_matrix_to_vector_of_appearence(matrix):
        return tf.reduce_sum(tf.unstack(matrix, axis=1), axis=1)


def cast_to_float32_and_binarize_pred(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_pred = y_pred * 2
    y_pred = tf.floor(y_pred)

    return y_true, y_pred


def compute_tp_fp_tn_fn(y_true, y_pred):
    y_true, y_pred = cast_to_float32_and_binarize_pred(y_true, y_pred)

    tp_matrix = y_pred * y_true

    fp_matrix = y_pred - tp_matrix

    fn_matrix = (y_pred + y_true) - 2 * tp_matrix - fp_matrix

    tn_matrix = 1 - (tp_matrix + fp_matrix + fn_matrix)

    tp = convert_from_matrix_to_vector_of_appearence(tp_matrix)
    fp = convert_from_matrix_to_vector_of_appearence(fp_matrix)
    tn = convert_from_matrix_to_vector_of_appearence(tn_matrix)
    fn = convert_from_matrix_to_vector_of_appearence(fn_matrix)

    return tp, fp, tn, fn



def first_class_f1_score(y_true, y_pred):
    tp, fp, tn, fn = compute_tp_fp_tn_fn(y_true, y_pred)

    f1 = 2 * tp / (2 * tp + fp + fn)
    
    return f1[0]


def second_class_f1_score(y_true, y_pred):
    tp, fp, tn, fn = compute_tp_fp_tn_fn(y_true, y_pred)

    f1 = 2 * tp / (2 * tp + fp + fn)

    return f1[1]

def third_class_f1_score(y_true, y_pred):
    tp, fp, tn, fn = compute_tp_fp_tn_fn(y_true, y_pred)

    f1 = 2 * tp / (2 * tp + fp + fn)

    return f1[2]

    

def mean_f1_score(y_true, y_pred):
    tp, fp, tn, fn = compute_tp_fp_tn_fn(y_true, y_pred)

    f1 = 2 * tp / (2 * tp + fp + fn)
    f1_mean = tf.reduce_mean(f1)

    return f1_mean


def first_class_precision(y_true, y_pred):
    tp, fp, tn, fn = compute_tp_fp_tn_fn(y_true, y_pred)

    return (tp / (tp + fp))[0]

def second_class_precision(y_true, y_pred):
    tp, fp, tn, fn = compute_tp_fp_tn_fn(y_true, y_pred)

    return (tp / (tp + fp))[1]


def third_class_precision(y_true, y_pred):
    tp, fp, tn, fn = compute_tp_fp_tn_fn(y_true, y_pred)

    return (tp / (tp + fp))[2]


def mean_precision(y_true, y_pred):
    tp, fp, tn, fn = compute_tp_fp_tn_fn(y_true, y_pred)

    return tf.reduce_mean(tp / (tp + fp))


def first_class_recall(y_true, y_pred):
    tp, fp, tn, fn = compute_tp_fp_tn_fn(y_true, y_pred)

    return (tp / (tp + fn))[0]


def second_class_recall(y_true, y_pred):
    tp, fp, tn, fn = compute_tp_fp_tn_fn(y_true, y_pred)

    return (tp / (tp + fn))[1]


def third_class_recall(y_true, y_pred):
    tp, fp, tn, fn = compute_tp_fp_tn_fn(y_true, y_pred)

    return (tp / (tp + fn))[2]


def mean_recall(y_true, y_pred):
    tp, fp, tn, fn = compute_tp_fp_tn_fn(y_true, y_pred)

    return tf.reduce_mean(tp / (tp + fn))


def first_class_accuracy(y_true, y_pred):
    tp, fp, tn, fn = compute_tp_fp_tn_fn(y_true, y_pred)

    return ((tn + tp) / (tp + fp + tn + fn))[0]



def second_class_accuracy(y_true, y_pred):
    tp, fp, tn, fn = compute_tp_fp_tn_fn(y_true, y_pred)

    return ((tn + tp) / (tp + fp + tn + fn))[1]

def third_class_accuracy(y_true, y_pred):
    tp, fp, tn, fn = compute_tp_fp_tn_fn(y_true, y_pred)

    return ((tn + tp) / (tp + fp + tn + fn))[2]

def mean_accuracy(y_true, y_pred):
    tp, fp, tn, fn = compute_tp_fp_tn_fn(y_true, y_pred)

    return tf.reduce_mean((tn + tp) / (tp + fp + tn + fn))

CUSTOM_METRICS = [
                    first_class_f1_score, second_class_f1_score, third_class_f1_score, 
                    mean_f1_score, first_class_precision, second_class_precision, third_class_precision, 
                    mean_precision, first_class_recall, second_class_recall, third_class_recall, mean_recall, 
                    first_class_accuracy, second_class_accuracy, third_class_accuracy, mean_accuracy]
