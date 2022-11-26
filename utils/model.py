import tensorflow as tf


def model_design(row, col, num_classes):

    tf.keras.initializers.GlorotNormal(241)
    input_layer = tf.keras.Input(
        shape=(
            row,
            col,
            1,
        )
    )
    layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(input_layer)
    layer = tf.keras.layers.GaussianNoise(0.1)(layer)

    # Block-1
    layer1 = tf.keras.layers.Conv1D(filters=16, kernel_size=3, activation="relu")(layer)
    layer1 = tf.keras.layers.BatchNormalization()(layer1)
    layer1 = tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation="relu")(
        layer1
    )
    layer1 = tf.keras.layers.BatchNormalization()(layer1)
    layer1 = tf.keras.layers.GaussianDropout(0.1)(layer1)

    # Block-2
    layer2 = tf.keras.layers.Conv1D(filters=48, kernel_size=7, activation="relu")(
        layer1
    )
    layer2 = tf.keras.layers.BatchNormalization()(layer2)
    layer2 = tf.keras.layers.Conv1D(filters=64, kernel_size=9, activation="relu")(
        layer2
    )
    layer2 = tf.keras.layers.BatchNormalization()(layer2)
    layer2 = tf.keras.layers.GaussianDropout(0.1)(layer2)

    lay2 = tf.keras.layers.LSTM(32, return_sequences=True)(layer2)
    lay2 = tf.keras.layers.Conv1DTranspose(
        filters=32, kernel_size=15, activation="relu"
    )(lay2)
    max1 = tf.keras.layers.MultiHeadAttention(
        num_heads=7,
        key_dim=2,
        activity_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.01),
    )(
        layer1 + lay2, layer1 + lay2
    )  # Reverse attention-1

    # Block-3
    layer3 = tf.keras.layers.Conv1D(filters=96, kernel_size=11, activation="relu")(
        layer2
    )
    layer3 = tf.keras.layers.BatchNormalization()(layer3)
    layer3 = tf.keras.layers.Conv1D(filters=128, kernel_size=13, activation="relu")(
        layer3
    )
    layer3 = tf.keras.layers.BatchNormalization()(layer3)
    layer3 = tf.keras.layers.GaussianDropout(0.1)(layer3)

    lay3 = tf.keras.layers.LSTM(32, return_sequences=True)(layer3)
    lay3 = tf.keras.layers.Conv1DTranspose(
        filters=64, kernel_size=23, activation="relu"
    )(lay3)
    max2 = tf.keras.layers.MultiHeadAttention(
        num_heads=10,
        key_dim=2,
        activity_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.01),
    )(
        layer2 + lay3, layer2 + lay3
    )  # Reverse Attention-2

    max1 = tf.reduce_max(max1 + layer1, 1)
    max2 = tf.reduce_max(max2 + layer2, 1)
    max3 = tf.reduce_max(layer3, 1)
    concat_layer = tf.keras.layers.concatenate([max1, max2, max3], 1)
    layer = tf.keras.layers.Dense(num_classes, activation="softmax")(concat_layer)

    model = tf.keras.Model(inputs=input_layer, outputs=layer)
    return model
