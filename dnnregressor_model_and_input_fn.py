import tensorflow as tf

# Function that generates and returns feature columns as expected by tf.estimators
def make_feature_columns_numeric(features = dict()):
    feature_columns = []
    for key in features.keys():
        feature_columns.append(tf.feature_column.numeric_column(key = key))
    return feature_columns

# Define input function expected by tf.estimator.train()
def train_input_fn(features, labels, batch_size, num_epochs = None):
    # Convert inputs to Dataset
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    # Shuffle repeat and batch samples
    dataset = dataset.shuffle(1000).repeat(num_epochs).batch(batch_size)

    # Return read end of pipeline
    return dataset.make_one_shot_iterator().get_next()

# Define input function expected by tf.estimator.eval()
def eval_input_fn(features, labels, batch_size, num_epochs = None):
    # Convert input to Dataset
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    dataset = dataset.repeat(num_epochs).batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()

# WRITE A MODEL FUNCTION
def dnn_regressor(features, # BATCH_FEATURES FROM input_fn
                 labels, # BATCH_LABELS from input_fn
                 mode, # TRAIN, PREDICT, EVAL
                 params):

    # DEFINING A MODEL
    # ----------------------
    l1_l2_reg = tf.contrib.layers.l1_l2_regularizer(
                scale_l1 = params['l_1'],
                scale_l2 = params['l_2'],
                scope = None
    )
    # Input layer
    net = tf.feature_column.input_layer(features,
                                        params['feature_columns'],
                                       )

    # Hidden layer
    cnt = 0
    for units in params['hidden_units']:
        # construct layer with correct activation functions
        if params['activations'][cnt] == 'relu':
            net = tf.layers.dense(
                inputs = net,
                units = units,
                activation = tf.nn.relu,
                kernel_regularizer = l1_l2_reg
            )
        if params['activations'][cnt] == 'sigmoid':
            net = tf.layers.dense(
                inputs = net,
                units = units,
                activation = tf.nn.sigmoid,
                kernel_regularizer = l1_l2_reg
            )
        cnt += 1


    # Output layer
    output = tf.layers.dense(inputs = net,
                             units = 1,
                             activation = tf.nn.relu # relu activation here to enforce positivity
                            )
    # ----------------------

    # PREDICTION MODE
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'output': output,
        }
        return tf.estimator.EstimatorSpec(mode = mode,
                                          predictions = predictions
                                         )

    # Compute loss
    if params['loss_fn'] == 'mse':
        loss = tf.losses.mean_squared_error(labels = labels,
                                            predictions = output
                                           ) + tf.losses.get_regularization_loss()

    if params['loss_fn'] == 'mae':
        loss = tf.losses.absolute_difference(labels = labels,
                                            predictions = output
                                            ) + tf.losses.get_regularization_loss()

    # TRAINING
    # Choose optimizer
    # ---------------------------------
    if params['optimizer'] == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = params['learning_rate'])

    if params['optimizer'] == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate = params['learning_rate'],
                                           beta1 = params['beta1'],
                                           beta2 = params['beta2'])
    if params['optimizer'] == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate = params['learning_rate'],
                                               rho = params['rho'])

    if params['optimizer'] == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate = params['learning_rate'],
                                               momentum = params['beta1'])


    if params['optimizer'] == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate = params['learning_rate'])
    # ----------------------------------

    # Training operation given optimizer
    train_op = optimizer.minimize(loss = loss,
                                  global_step = tf.train.get_global_step()
                                 )

    # Compute evaluation metrics of interest
    mse = tf.metrics.mean_squared_error(labels = labels,
                                        predictions = tf.cast(output, tf.float64)
                                       )

    mae = tf.metrics.mean_absolute_error(labels = labels,
                                         predictions = tf.cast(output, tf.float64)
                                        )
    # Store evaluation metric to be returned in mode = 'eval'
    metrics = {'mse': mse,
               'mae': mae
               }

    # Data to be reported for tensorboard to access
    tf.summary.scalar('loss', loss)

    # EVALUATION MODE
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode = mode,
                                          loss = loss,
                                          eval_metric_ops = metrics
                                         )

    return tf.estimator.EstimatorSpec(mode = mode,
                                      loss = loss,
                                      train_op = train_op
                                     )
