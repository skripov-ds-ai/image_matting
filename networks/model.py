from tensorflow.keras.layers import UpSampling2D, DepthwiseConv2D, ReLU, Conv2D, Convolution2D, Softmax, Add, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.regularizers import L2


def create_model(opt, metrics, loss, trainable_pretrained=True):
    old_model = MobileNet(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
    old_model.trainable = trainable_pretrained

    x = old_model.output
    y_names = ["conv_pw_11_relu", "conv_pw_5_relu", "conv_pw_3_relu", "conv_pw_1_relu"]
    f_nums = [1024, 64, 64, 64]
    ys = [
        Conv2D(f_num, kernel_size=1, name=f'skip_hair_conv_{i}')(old_model.get_layer(name=name).output)
        for i, (name, f_num) in enumerate(zip(y_names, f_nums))
    ] + [None]

    for i in range(5):
        y = ys[i]
        x = UpSampling2D(name=f'upsampling_hair_{i}')(x)
        if y is not None:
            x = Add(name=f'skip_hair_add_{i}')([x, y])
        x = DepthwiseConv2D(
            kernel_size=3,
            padding='same',
            name=f'depth_conv2d_hair_{i}',
            kernel_initializer=GlorotNormal(seed=(i + 1)),
        )(x)
        x = Conv2D(
            64,
            kernel_size=1,
            padding='same',
            name=f'conv2d_hair_{i}',
            kernel_regularizer=L2(2e-5),
            kernel_initializer=GlorotNormal(seed=11*(i + 1)),
        )(x)
        x = ReLU(name=f'relu_hair_{i}')(x)
    x = Conv2D(
        1,
        # 2,
        kernel_size=1,
        padding='same',
        name='conv2d_hair_final',
        kernel_regularizer=L2(2e-5),
        kernel_initializer=GlorotNormal(seed=0)
    )(x)
    # x = Softmax(name='sigmoid_hair_final')(x)
    x = Activation('sigmoid', name='sigmoid_hair_final')(x)

    model = Model(old_model.input, x)
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=metrics,
    )
    return model

