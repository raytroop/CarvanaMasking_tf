"""Define the model."""

import tensorflow as tf

def conv_block(input_tensor, num_filters, is_training):
    x = tf.layers.conv2d(input_tensor, filters=num_filters, kernel_size=(3, 3), padding='same')
    x = tf.layers.batch_normalization(x, trainable=True, training=is_training)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, filters=num_filters, kernel_size=(3, 3), padding='same')
    x = tf.layers.batch_normalization(x, trainable=True, training=is_training)
    x = tf.nn.relu(x)
    return x

def encoder_block(input_tensor, num_filters, is_training):
    x = conv_block(input_tensor, num_filters, is_training)
    x_pool = tf.layers.max_pooling2d(x, (2, 2), (2, 2))
    return x_pool, x

def decoder_block(input_tensor, concat_tensor, num_filters, is_training):
    x = tf.layers.conv2d_transpose(input_tensor, filters=num_filters, kernel_size=(2, 2), strides=(2, 2), padding='same')
    x = tf.concat([concat_tensor, x], axis=-1)
    x = tf.layers.batch_normalization(x, trainable=True, training=is_training)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, filters=num_filters, kernel_size=(3, 3), padding='same')
    x = tf.layers.batch_normalization(x, trainable=True, training=is_training)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, filters=num_filters, kernel_size=(3, 3), padding='same')
    x = tf.layers.batch_normalization(x, trainable=True, training=is_training)
    x = tf.nn.relu(x)
    return x


def build_model(inputs, is_training):
    with tf.name_scope('encoder'):
        # 256
        encoder0_pool, encoder0 = encoder_block(inputs, 32, is_training)
        # 128
        encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64, is_training)
        # 64
        encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128, is_training)
        # 32
        encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256, is_training)
        # 16
        encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512, is_training)
        # 8
        center = conv_block(encoder4_pool, 1024, is_training)
    with tf.name_scope('decoder'):
        # center
        decoder4 = decoder_block(center, encoder4, 512, is_training)
        # 16
        decoder3 = decoder_block(decoder4, encoder3, 256, is_training)
        # 32
        decoder2 = decoder_block(decoder3, encoder2, 128, is_training)
        # 64
        decoder1 = decoder_block(decoder2, encoder1, 64, is_training)
        # 128
        decoder0 = decoder_block(decoder1, encoder0, 32, is_training)
        # 256
        logits = tf.layers.conv2d(decoder0, filters=1, kernel_size=(1, 1), activation=None)
    return logits

def dice_coeff(y_true, y_pred):
    """
    Dice coefficient

    Arguments:
        y_true: true mask 0/1
        y_pred: logits,  same shape and type with y_true
    """
    smooth = 1.
    # Flatten
    y_pred = tf.nn.sigmoid(y_pred)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    """
    Dice loss is a metric that measures overlap.

    Arguments:
        y_true: true mask 0/1
        y_pred: logits,  same shape and type with y_true
    """
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def sigmoid_dice_loss(y_true, y_pred):
    """"
    Here, we'll use a specialized loss function that combines binary cross entropy and our dice loss.
    This is based on individuals who competed within this competition obtaining better results empirically.

    Arguments:
        y_true: true mask 0/1
        y_pred: logits,  same shape and type with y_true
    """

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)) + dice_loss(y_true, y_pred)
    return loss

def model_fn(mode, inputs, params, reuse=False):
    """Model function defining the graph operations.

    Args:
        mode: (string) can be 'train' or 'eval'
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    labels = inputs['labels']

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        logits = build_model(inputs['images'], is_training)
        predictions = tf.cast(logits > 0.0, tf.float32)

    # Define loss and accuracy
    loss = sigmoid_dice_loss(labels, logits)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))

    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        if params.use_batch_norm:
            # Add a dependency to update the moving mean and variance for batch normalization
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer.minimize(loss, global_step=global_step)
        else:
            train_op = optimizer.minimize(loss, global_step=global_step)

    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        metrics = {
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions),
            'loss': tf.metrics.mean(loss),
            'mean_iou': tf.metrics.mean_iou(labels=tf.to_int64(labels), predictions=tf.to_int64(predictions), num_classes=2)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    summs = []
    summs.append(tf.summary.scalar('loss', loss))
    summs.append(tf.summary.scalar('accuracy', accuracy))
    summs.append(tf.summary.image('image_true', inputs['images']))
    summs.append(tf.summary.image('mask_true', inputs['labels']))
    summs.append(tf.summary.image('mask_preds', predictions))

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec["predictions"] = predictions
    model_spec['loss'] = loss
    model_spec['accuracy'] = accuracy
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    # `tf.summary.merge_all` is high risk, which collect all pre-created summary
    # eval will contain train, resulting OutOfRangeError(train data running out)
    model_spec['summary_op'] = tf.summary.merge(summs)

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec
