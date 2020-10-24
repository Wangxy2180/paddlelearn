import os
import random
import paddle.fluid as fluid
import cv2
import numpy as np


class Transform(object):
    def __init__(self, size=256):
        self.size = size

    def __call__(self, input, label):
        input = cv2.resize(input, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        return input, label


class BasicDataLoader():
    def __init__(self, image_folder, image_list_file, transform=None, shuffle=True):
        self.image_folder = image_folder
        self.image_list_file = image_list_file
        self.transform = transform
        self.shuffle = shuffle

        self.data_list = self.read_list()
        # print(self.data_list)

    def read_list(self):
        # read all image
        data_list = []
        with open(self.image_list_file) as infile:
            # lines = infile.readlines()
            for line in infile:
                data_path = os.path.join(self.image_folder, line.split()[0])
                label_path = os.path.join(self.image_folder, line.split()[1])
                data_list.append((data_path, label_path))
        random.shuffle(data_list)
        return data_list

    def preprocess(self, data, label):
        h, w, c = data.shape
        h_gt, w_gt = label.shape
        assert h == h_gt, "Error"
        assert w == w_gt, "Error"

        if self.transform:
            data, label = self.transform(data, label)

        label = label[:, :, np.newaxis]
        return data, label

    def __len__(self):
        return len(self.data_list)

    def __call__(self, *args, **kwargs):
        for data_path, label_path in self.data_list:
            data = cv2.imread(data_path)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            print(data.shape, label.shape)
            data, label = self.preprocess(data, label)
            # important!!
            yield data, label


def main():
    batch_size = 5
    place = fluid.CPUPlace()  # run in cpu 在cpu上跑程序
    with fluid.dygraph.guard(place):
        transform = Transform(256)
        # 创建一个dataloader instance
        # transform is a 256*256 object
        basic_dataloader = BasicDataLoader(
            image_folder='./dummy_data',
            image_list_file='../dummy_data/list.txt',
            transform=transform,
            shuffle=True
        )
        # create fluid.io instance
        # capacity is the num of batch

        dataloader = fluid.io.DataLoader.from_generator(capacity=1, use_multiprocess=False)
        # pack a batch into a tensor
        # dataloader'data is same as basic_dataloader return

        # !!!!data_loader return a 4-d NWHC tensor,so the image from basic_dataloader must be equal in W and H
        dataloader.set_sample_generator(basic_dataloader,
                                        batch_size=batch_size,
                                        places=place)
        # set sample generator for fluid dataloader

        num_epoch = 2
        for epoch in range(1, num_epoch + 1):
            print(f'Epoch [{epoch}/{num_epoch}]:')
            # dataloader'sample_generator need return (data, label)
            for idx, (data, label) in enumerate(dataloader):
                print(f'Iter {idx},Data shape:{data.shape},label shape:{label.shape}')


if __name__ == "__main__":
    main()
