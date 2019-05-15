import tensorflow as tf


def hash_loss(hash_code, labels, hash_code_num=16, batch_size=32):
    hash_point = {}
    hash_code = tf.squeeze(hash_code, [1, 2], name='hash_code/squeezed')
    
    hash_code_out = tf.matmul(hash_code, hash_code, transpose_b=True)
    hash_code_out_diag = tf.diag_part(hash_code_out)

    
    labels = tf.to_float(labels)
    print('---------------------labels', labels)
    y_s = tf.sign(tf.matmul(labels, labels, transpose_b=True) - 0.5)
    y_in = y_s - tf.diag(tf.diag_part(y_s))

    
    hash_distance = 0.5 * (hash_code_num - hash_code_out)
    hash_distance_S = hash_distance * y_in
    hash_point['hash_distance'] = hash_distance_S

    print('hash_code_num:', hash_code_num)
    
    hash_distance_mean = tf.reduce_sum(hash_distance_S, name='hash_distance_sum') / tf.cast(
        batch_size * (batch_size - 1), dtype=tf.float32)
    tf.add_to_collection('hash_distance_mean_losses', hash_distance_mean / hash_code_num)

    hash_max = tf.reduce_max(hash_distance_S) / hash_code_num
    hash_point['hash_max'] = hash_max
    hash_min = tf.reduce_max(
        tf.where(tf.less(hash_distance_S, 0), hash_distance_S,
                 tf.constant(-float(hash_code_num), shape=hash_distance_S.get_shape()))) / hash_code_num
    hash_point['hash_min'] = hash_min

    λ1 = 1
    λ2 = 1
    tf.add_to_collection('hash_distance_max_losses', hash_max * λ1)

    tf.add_to_collection('hash_distance_min_losses', hash_min * λ1)

    
    hash_code_out_diag_mean = 1 - tf.reduce_mean(hash_code_out_diag) / hash_code_num
    tf.add_to_collection('hash_code_diag_mean_losses', hash_code_out_diag_mean * λ2)

    return hash_point
