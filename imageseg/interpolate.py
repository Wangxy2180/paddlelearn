import paddle.fluid as fluid
import numpy as np


def main():
    with fluid.dygraph.guard(fluid.CPUPlace()):
        # data = np.array([[1, 2],
        #                  [3, 4]]).astype(np.float32)
        data = np.array([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]]).astype(np.float32)
        # reshape array to NCHW, input tensor should be 4-d
        data = data[np.newaxis, np.newaxis, :, :]
        # put data in cpu
        data = fluid.dygraph.to_variable(data)

        # out = fluid.layers.interpolate(data, out_shape=(4, 4), align_corners=True)
        out = fluid.layers.interpolate(data, out_shape=(6, 6), align_corners=True)
        # tack back data from cpu
        out = out.numpy()
        # remove 2 newaxis
        print(out)
        print(out.squeeze((0, 1)))


if __name__ == '__main__':
    main()
