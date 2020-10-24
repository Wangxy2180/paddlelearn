import cv2
import numpy as np
import os


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label=None):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class Normalize(object):
    def __init__(self, mean_val, std_val, val_scale=1):
        # set val_scale = 1 if mean and std are in range (0,1)
        # set val_scale to other value, if mean and std are in range (0,255)
        self.mean = np.array(mean_val, dtype=np.float32)
        self.std = np.array(std_val, dtype=np.float32)
        self.val_scale = 1 / 255.0 if val_scale == 1 else 1

    def __call__(self, image, label=None):
        image = image.astype(np.float32)
        image = image * self.val_scale
        image = image - self.mean
        image = image * (1 / self.std)
        return image, label


class ConvertDataType(object):
    def __call__(self, image, label=None):
        if label is not None:
            label = label.astype(np.int64)
        return image.astype(np.float32), label


class Pad(object):
    def __init__(self, size, ignore_label=255, mean_val=0, val_scale=1):
        # set val_scale to 1 if mean_val is in range (0, 1)
        # set val_scale to 255 if mean_val is in range (0, 255)
        factor = 255 if val_scale == 1 else 1

        self.size = size
        self.ignore_label = ignore_label
        self.mean_val = mean_val
        # from 0-1 to 0-255
        if isinstance(self.mean_val, (tuple, list)):
            self.mean_val = [int(x * factor) for x in self.mean_val]
        else:
            self.mean_val = int(self.mean_val * factor)

    def __call__(self, image, label=None):
        h, w, c = image.shape
        pad_h = max(self.size - h, 0)
        pad_w = max(self.size - w, 0)

        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)

        if pad_h > 0 or pad_w > 0:

            image = cv2.copyMakeBorder(image,
                                       top=pad_h_half,
                                       left=pad_w_half,
                                       bottom=pad_h - pad_h_half,
                                       right=pad_w - pad_w_half,
                                       borderType=cv2.BORDER_CONSTANT,
                                       value=self.mean_val)
            if label is not None:
                label = cv2.copyMakeBorder(label,
                                           top=pad_h_half,
                                           left=pad_w_half,
                                           bottom=pad_h - pad_h_half,
                                           right=pad_w - pad_w_half,
                                           borderType=cv2.BORDER_CONSTANT,
                                           value=self.ignore_label)
        return image, label


# TODO
class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, label=None):
        h, w, c = image.shape
        # center = (x,y)
        center = (int(w / 2), int(h / 2))
        # crop a square, center = (x,y)
        left_top = (center[0] - int(self.size / 2), center[1] - int(self.size / 2))
        right_bottom = (center[0] + int(self.size / 2), center[1] + int(self.size / 2))
        out_image = image[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0], :]
        out_label = label[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]]
        return out_image, out_label


class CenterCrop_2(object):
    def __init__(self, crop_size):
        self.crop_h = crop_size
        self.crop_w = crop_size

    def __call__(self, image, label=None):
        h, w, c = image.shape
        top = (h - self.crop_h) // 2
        left = (h - self.crop_w) // 2
        image = image[top:top + self.crop_h, left:left + self.crop_w, :]
        if label is not None:
            label = label[top:top + self.crop_h, left:left + self.crop_w]
        return image, label


# TODO
class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, label=None):
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        return image, label


# TODO
class RandomFlip(object):
    def __init__(self):
        # [1,0,-1]=['horizon','vertival','both']
        # self.flip_type = ['horizon', 'vertical']
        # randint is [-1,2)
        self.flip_type = np.random.randint(0, 2)

    def __call__(self, image, label=None):
        h, w, c = image.shape
        out_image = cv2.flip(image, self.flip_type)
        out_label = cv2.flip(label, self.flip_type)
        return out_image, out_label


class RandomFlip_2(object):
    def __call__(self, image, label=None):
        prob_of_flip = np.random.rand()
        if prob_of_flip > 0.5:
            image = cv2.flip(image, 1)
            if label is not None:
                label = cv2.flip(label, 1)


# TODO
class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, label=None):
        h, w, c = image.shape
        # center is x_rand,y_rand
        x_rand = np.random.randint(int(self.size / 2), w - int(self.size / 2))
        y_rand = np.random.randint(int(self.size / 2), h - int(self.size / 2))
        left_top = (x_rand - int(self.size / 2), y_rand - int(self.size / 2))
        right_bottom = (x_rand + int(self.size / 2), y_rand + int(self.size / 2))
        out_image = image[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0], :]
        out_label = label[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]]

        return out_image, out_label


class RandomCrop_2(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, image, label=None):
        h, w, c = image.shape
        top = np.random.uniform(h - self.crop_size)
        left = np.random.uniform(w - self.crop_size)
        assert top >= 0, "Error: crop_siize > image height !"
        assert left >= 0, "Error: crop_siize > image width !"
        rect = np.array([int(left),
                         int(top),
                         int(left + self.crop_size),
                         int(top + self.crop_size)])
        image = image[rect[1]:rect[3], rect[0]:rect[2], :]
        if label is not None:
            label = label[rect[1]:rect[3], rect[0]:rect[2]]
        return image, label


# TODO
class Scale(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, image, label=None):
        h, w, c = image.shape
        out_image = cv2.resize(image, (h * int(self.scale), w * int(self.scale / 2)))
        out_label = cv2.resize(label, (h * int(self.scale), w * int(self.scale / 2)))
        return out_image, out_label


class Scale_2:
    def __call__(self, image, label=None, scale=1.0):
        if not isinstance(scale, (list, tuple)):
            scale = [scale, scale]
        h, w, c = image.shape
        image = cv2.resize(image,
                           (int(w * scale[0]), int(h * scale[1])),
                           interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label,
                               (int(w * scale[0]), int(h * scale[1])),
                               interpolation=cv2.INTER_NEAREST)
        return image, label


# TODO
class RandomScale(object):
    def __init__(self):
        self.scale = round(np.random.uniform(0, 2), 2)

    def __call__(self, image, label=None):
        h, w, c = image.shape
        out_image = cv2.resize(image, (h * int(self.scale), w * int(self.scale / 2)))
        out_label = cv2.resize(label, (h * int(self.scale), w * int(self.scale / 2)))
        return out_image, out_label


class RandomScale_2:
    def __init__(self, min_scale=0.5, max_scale=2.0, step=0.25):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.step = step
        self.scale = Scale()

    def __call__(self, image, label=None):
        if self.step == 0:
            self.random_scale = np.random.uniform(self.min_scale, self.max_scale, 1)[0]
        else:
            num_steps = int((self.max_scale - self.min_scale) / self.step + 1)
            scale_factors=np.linspace(self.min_scale,self.max_scale,num_steps)
            np.random.shuffle(scale_factors)
            self.random_scale=scale_factors[0]
        image,label = self.scale(image,label,self.random_scale)

        return image,label


def main():
    image = cv2.imread('../dummy_data/JPEGImages/2008_000064.jpg')
    label = cv2.imread('../dummy_data/GroundTruth_trainval_png/2008_000064.png')

    # TODO: crop_size
    crop_size = 256
    # TODO: Transform: RandomSacle, RandomFlip, Pad, RandomCrop
    transform = RandomCrop(size=crop_size)
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save')
    print('save_path is ', save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(10):
        transformed_image, transformed_label = transform(image, label)
        cv2.imwrite(os.path.join(save_path, '{}.jpg'.format(i)), transformed_image)
        cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), transformed_label)
        print(f'transfrom_{i} have saved')

def main_2():
    image = cv2.imread('../dummy_data/JPEGImages/2008_000064.jpg')
    label = cv2.imread('../dummy_data/GroundTruth_trainval_png/2008_000064.png')
    crop_size=256
    augment = Compose([
        RandomScale_2(),
        RandomFlip_2(),
        Pad(crop_size,mean_val=[0.485,0.456,0.406]),
        RandomCrop_2(crop_size),
        ConvertDataType(),
        Normalize(0,1)])
    new_img,new_label = augment(image,label)
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save')
    print('save_path is ', save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cv2.imwrite(os.path.join(save_path, 'test.jpg'), new_img)
    cv2.imwrite(os.path.join(save_path, 'test.png'), new_label)

# TODO: call transform
# TODO: save image


if __name__ == "__main__":
    main()
    # print(os.path.dirname(os.path.abspath(__file__)))
