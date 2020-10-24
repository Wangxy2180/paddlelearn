import paddle
import paddle.fluid as fluid
import numpy as np
from paddle.fluid.dygraph import to_variable
from paddle.fluid.dygraph import Conv2D
from paddle.fluid.dygraph import Pool2D
np.set_printoptions(precision=2)

class BasicModel(fluid.dygraph.Layer):
    def __init__(self,num_classes=59):
        super(BasicModel,self).__init__()
        # self.pool = Pool2D(pool_size=2,pool_stride=2)
        self.pool = Pool2D(pool_size=2,pool_stride=2)
        # self.conv = Conv2D(num_channels=3, num_filters=1,filter_size=1)
        self.conv = Conv2D(num_channels=3, num_filters=num_classes, filter_size=1)

    # why pool is first:wq
    def forward(self, inputs):
        x = self.pool(inputs)
        # interpolate,default data_format='NCHW',or 'NHWC' for5-D,NCDHW(depth)
        # input is 4-D tensor, out_shape is math(out_h,out_w)
        x = fluid.layers.interpolate(x,out_shape=inputs.shape[2::])
        x = self.conv(x)
        return x

def main():
    place = paddle.fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        model = BasicModel(num_classes=59)
        model.eval()#预测模式，train训练模式
        input_data = np.random.rand(1,3,8,8).astype(np.float32)# 数据一般是4维，nchw
        print("input data shape",input_data.shape)
        input_data = to_variable(input_data) # numpy转paddletensor
        output_data = model(input_data)
        output_data = output_data.numpy() # 当作一个numpyarray
        print("output daya shape",output_data.shape)

if __name__ == '__main__':
    main()