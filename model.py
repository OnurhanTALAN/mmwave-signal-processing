import tensorflow as tf
from tensorflow.keras import layers, models, Model

class AttentionLayer(layers.Layer):
    """
    Attention mechanism for MIL.
    Computes attention weights for each instance in a bag.
    """
    def __init__(self, attention_dim=128, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.attention_dim = attention_dim
    
    def build(self, input_shape):
        # input_shape: (batch_size, num_instances, feature_dim)
        feature_dim = input_shape[-1]
        
        # Attention weights
        self.W = self.add_weight(
            name='attention_W',
            shape=(feature_dim, self.attention_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_b',
            shape=(self.attention_dim,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_u',
            shape=(self.attention_dim, 1),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        # inputs shape: (batch_size, num_instances, feature_dim)
        # Compute attention scores
        # e_ij = u^T * tanh(W * h_j + b)
        attention_score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        attention_score = tf.tensordot(attention_score, self.u, axes=1)
        
        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(attention_score, axis=1)
        
        # Weighted sum of instances
        weighted_features = inputs * attention_weights
        bag_representation = tf.reduce_sum(weighted_features, axis=1)
        
        return bag_representation, attention_weights
    
    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({'attention_dim': self.attention_dim})
        return config


def create_instance_encoder(input_shape=(128, 128, 1), dropout_rate=0.5):
    """
    Creates a CNN-based feature extractor for individual instances.
    
    Args:
        input_shape: Shape of a single instance (spectrogram)
        dropout_rate: Dropout rate for regularization
    
    Returns:
        Feature encoder model
    """
    instance_input = layers.Input(shape=input_shape, name='instance_input')
    
    # Convolutional blocks for feature extraction
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1')(instance_input)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)
    x = layers.Dropout(dropout_rate * 0.5, name='dropout1')(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)
    x = layers.Dropout(dropout_rate * 0.5, name='dropout2')(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.MaxPooling2D((2, 2), name='pool3')(x)
    x = layers.Dropout(dropout_rate * 0.75, name='dropout3')(x)
    
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4')(x)
    x = layers.BatchNormalization(name='bn4')(x)
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    
    # Dense layers for instance-level features
    x = layers.Dense(256, activation='relu', name='fc1')(x)
    x = layers.BatchNormalization(name='bn5')(x)
    x = layers.Dropout(dropout_rate, name='dropout4')(x)
    instance_features = layers.Dense(128, activation='relu', name='instance_features')(x)
    
    encoder = Model(inputs=instance_input, outputs=instance_features, name='instance_encoder')
    return encoder


def create_model(input_shape=(128, 128, 1), num_instances=10, dropout_rate=0.5, attention_dim=128):
    """
    Create an Attention-based MIL model for radar spectrogram classification.
    
    Args:
        input_shape: Shape of a single instance/spectrogram (height, width, channels)
        num_instances: Number of instances in each bag
        dropout_rate: Dropout rate for regularization
        attention_dim: Dimensionality of attention mechanism
    
    Returns:
        MIL model with attention mechanism
    """
    # Input: a bag of instances
    bag_input = layers.Input(shape=(num_instances,) + input_shape, name='bag_input')
    
    # Create instance encoder
    instance_encoder = create_instance_encoder(input_shape=input_shape, dropout_rate=dropout_rate)
    
    # Apply encoder to each instance using TimeDistributed
    instance_features = layers.TimeDistributed(instance_encoder, name='encode_instances')(bag_input)
    
    # Apply attention mechanism
    attention_layer = AttentionLayer(attention_dim=attention_dim, name='attention')
    bag_features, attention_weights = attention_layer(instance_features)
    
    # Bag-level classification
    x = layers.Dense(64, activation='relu', name='bag_fc1')(bag_features)
    x = layers.Dropout(dropout_rate, name='bag_dropout')(x)
    output = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    # Create model
    model = Model(inputs=bag_input, outputs=output, name='attention_mil_model')
    
    return model
