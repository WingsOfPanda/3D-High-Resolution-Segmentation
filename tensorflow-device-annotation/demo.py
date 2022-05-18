import tensorflow as tf
from tensorflow.keras import layers, Input, Model

tf.random.set_seed(1)

# if additional flags are needed, define it here.
gpus = tf.config.experimental.list_physical_devices('GPU')

# Currently, memory growth needs to be the same across GPUs
for gpu in gpus:
    print(gpu)
    tf.config.experimental.set_memory_growth(gpu, True)

input_size = (256, 256, 256)

input_tensor = tf.ones((1,) + input_size + (1,), dtype=tf.float32)

ft = 24


def part1(it):
    x = layers.Conv3D(ft, 3, padding='same')(it)
    x = layers.Conv3D(ft, 5, padding='same')(x)

    ot0 = layers.Conv3D(ft, 7, padding='same')(x)
    ot1 = layers.MaxPool3D(pool_size=2)(ot0)

    return Model(it, outputs=[ot0, ot1], name='part1')


def part2(it0, it1):
    x = layers.Conv3D(ft, 3, padding='same')(it1)
    x = layers.UpSampling3D(size=2)(x)
    x = layers.concatenate([x, it0], axis=-1)
    x = layers.Conv3D(ft, 5, padding='same')(x)
    x = layers.Conv3D(ft, 7, padding='same')(x)
    ot = layers.Conv3D(1, 3, padding='same')(x)
    return Model(inputs=[it0, it1], outputs=[ot], name='part2')


it1_ = Input(shape=input_size + (1,))

it2_0_ = Input(shape=input_size + (ft,))
it2_1_ = Input(shape=tuple([int(i/2) for i in list(input_size)]) + (ft, ))
print(it2_0_.shape, it2_1_.shape)


Part1 = part1(it1_)
Part2 = part2(it2_0_, it2_1_)

Part1.summary(line_length=128)
Part2.summary(line_length=128)

print(f'tf version: {tf.__version__}')

with tf.GradientTape() as tape:
    with tf.device('/gpu:0'):
        x0_, x1_ = Part1(input_tensor)
        print(f'forward part1 done', flush=True)

    with tf.device('/gpu:1'):
        ot_, = Part2([x0_, x1_])
        print(f'forward part2 done', flush=True)

    g = tape.gradient(ot_, Part2.trainable_variables + Part1.trainable_variables)
    print(f'backward part1&2&3&4 done', flush=True)
    print(g[0][0])

# with tf.GradientTape() as tape:
#     x0_, x1_ = Part1(input_tensor)
#     print(f'forward part1 done', flush=True)
#
#     ot_, = Part2([x0_, x1_])
#     print(f'forward part2 done', flush=True)
#
#     g0 = tape.gradient(ot_, Part2.trainable_variables + Part1.trainable_variables)
#     print(f'backward done', flush=True)
#
#     print(g0[0][0])

