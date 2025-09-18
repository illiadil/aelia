import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate, Dense, LayerNormalization, Add, Reshape, Lambda, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
import tensorflow.keras.backend as K
import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, Conv2DTranspose, Concatenate, BatchNormalization, UpSampling2D
from keras.layers import  Dropout, Activation
from keras.optimizers import Adam, SGD

def transformer_block(inputs, num_heads=8, ff_dim=2048, dropout_rate=0.1):
    """Simplified Vision Transformer block using Keras layers."""
    # Get input shape
    batch_size, height, width, channels = inputs.shape

    # Reshape to (batch, height*width, channels) for attention
    x = Reshape((height * width, channels))(inputs)

    # Multi-Head Self-Attention
    attention_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=channels)(x, x)
    attention_output = tf.keras.layers.Dropout(dropout_rate)(attention_output)
    out1 = LayerNormalization(epsilon=1e-6)(Add()([x, attention_output]))

    # Feed-Forward Network
    ffn = tf.keras.Sequential([
        Dense(ff_dim, activation='relu'),
        Dense(channels),
        tf.keras.layers.Dropout(dropout_rate)
    ])
    ffn_output = ffn(out1)
    out2 = LayerNormalization(epsilon=1e-6)(Add()([out1, ffn_output]))

    # Reshape back to (batch, height, width, channels)
    out2 = Reshape((height, width, channels))(out2)
    return out2

def transunet(sz=(256, 256, 3)):
    # Input
    inputs = Input(sz)

    # Encoder: ResNet-50 backbone (pre-trained)
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
    # Extract feature maps at different scales (C1, C2, C3, C4, C5)
    layers = [
        base_model.get_layer('conv1_relu').output,           # 128x128x64
        base_model.get_layer('conv2_block3_out').output,     # 64x64x256
        base_model.get_layer('conv3_block4_out').output,     # 32x32x512
        base_model.get_layer('conv4_block6_out').output,     # 16x16x1024
        base_model.get_layer('conv5_block3_out').output      # 8x8x2048
    ]

    # Bottleneck: Vision Transformer
    x = layers[-1]  # Shape: (8, 8, 2048)
    x = transformer_block(x, num_heads=8, ff_dim=2048)  # Apply transformer

    # Decoder: Upsampling with skip connections
    f = 2048  # Initial filters from bottleneck
    ff2 = 512  # Filters for transpose convolution
    skip_layers = layers[::-1][1:]  # Reverse skip connections (C4, C3, C2, C1)
    j = 0 # Start from the first skip connection (C4)

    # First upsampling
    x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same')(x)  # 16x16x512
    # Upsample skip connection C4 to match the transposed convolution output shape
    skip_upsampled = UpSampling2D(size=(16//skip_layers[j].shape[1], 16//skip_layers[j].shape[2]))(skip_layers[j])
    x = Concatenate(axis=3)([x, skip_upsampled])  # Concat with upsampled C4
    j += 1
    f = ff2
    x = Conv2D(f, 3, activation='relu', padding='same')(x)
    x = Conv2D(f, 3, activation='relu', padding='same')(x)

    # Remaining upsampling steps
    for i in range(3):  # For C3, C2, C1
        ff2 = ff2 // 2
        f = f // 2
        x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same')(x)
        # Upsample skip connection to match the transposed convolution output shape
        skip_upsampled = UpSampling2D(size=(x.shape[1]//skip_layers[j].shape[1], x.shape[2]//skip_layers[j].shape[2]))(skip_layers[j])
        x = Concatenate(axis=3)([x, skip_upsampled])
        j += 1
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        x = Conv2D(f, 3, activation='relu', padding='same')(x)

    # Final upsampling to match input size
    x = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(x)  # 256x256x64
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)

    # Classification
    outputs = Conv2D(1, 1, activation='sigmoid')(x)

    # Model creation
    model = Model(inputs=[inputs], outputs=[outputs])

    # Combined loss: Binary Cross-Entropy + Dice Loss
    def bce_dice_loss(y_true, y_pred):
        bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        dice = (2. * intersection + 1.) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.)
        return bce + (1 - dice)

    model.compile(optimizer='adam', loss=bce_dice_loss, metrics=['accuracy'])

    return model

def unet(sz = (256, 256, 3)):
  x = Input(sz)
  inputs = x

  #down sampling
  f = 8
  layers = []

  for i in range(0, 6):
    x = Conv2D(f, 3, activation='relu', padding='same') (x)
    x = Conv2D(f, 3, activation='relu', padding='same') (x)
    layers.append(x)
    x = MaxPooling2D() (x)
    f = f*2
  ff2 = 64

  #bottleneck
  j = len(layers) - 1
  x = Conv2D(f, 3, activation='relu', padding='same') (x)
  x = Conv2D(f, 3, activation='relu', padding='same') (x)
  x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same') (x)
  x = Concatenate(axis=3)([x, layers[j]])
  j = j -1

  #upsampling
  for i in range(0, 5):
    ff2 = ff2//2
    f = f // 2
    x = Conv2D(f, 3, activation='relu', padding='same') (x)
    x = Conv2D(f, 3, activation='relu', padding='same') (x)
    x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same') (x)
    x = Concatenate(axis=3)([x, layers[j]])
    j = j -1


  #classification
  x = Conv2D(f, 3, activation='relu', padding='same') (x)
  x = Conv2D(f, 3, activation='relu', padding='same') (x)
  outputs = Conv2D(1, 1, activation='sigmoid') (x)

  #model creation
  model = Model(inputs=[inputs], outputs=[outputs])
  model.compile(optimizer = 'adam', loss = 'binary_crossentropy')

  return model


# Create models
transunet_model = transunet()
unet_model = unet()
