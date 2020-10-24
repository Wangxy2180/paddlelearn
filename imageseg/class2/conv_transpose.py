import paddle.fluid as fluid
from paddle.fluid import ParamAttr
import numpy as np

np.set_printoptions(precision=2)


def main():
    with fluid.dygraph.guard(fluid.CPUPlace()):
        data = np.array([[1, 2],
                         [3, 4]]).astype(np.float32)
        print(data)
        # reshape data NCHW
        data = data[np.newaxis, np.newaxis, :, :]

        data = fluid.dygraph.to_variable(data)

        # preset conv weight
        convw = np.array([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]]).astype(np.float32)
        print(convw)
        convw = convw[np.newaxis, np.newaxis, :, :]
        w_init = fluid.initializer.NumpyArrayInitializer(convw)
        param_attr = fluid.ParamAttr(initializer=w_init)
        conv_t = fluid.dygraph.Conv2DTranspose(num_channels=1,
                                               num_filters=1,
                                               filter_size=3,
                                               padding=0,
                                               stride=1,
                                               param_attr=param_attr)
        # inference
        out = conv_t(data)
        out = out.numpy()

        print(out.squeeze((0, 1)))


if __name__ == '__main__':
    main()
