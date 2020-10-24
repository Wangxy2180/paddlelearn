import os
import paddle
import paddle.fluid as fluid
from paddle.fluid.optimizer import AdamOptimizer
import numpy as np
import argparse
from utils import AverageMeter
from class1.basic_model import BasicModel
from basic_seg_loss import Basic_SegLoss
from basic_data_preprocessing import TrainAugmentation

parser = argparse.ArgumentParser()
parser.add_argument('--net', type=str, default='basic')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--image_folder', type=str, default='./dummy_data')
parser.add_argument('--image_list_file', type=str, default='./dummy_data/list.txt')
parser.add_argument('--checkpoint_folder', type=str, default='./output')
parser.add_argument('--save_freq', type=int, default=2)

args = parser.parse_args()


def train(dataloader, model, criterion, optimizer, epoch, total_batch):
    model.train() # change to train mode
    train_loss_meter = AverageMeter()
    for batch_id, data in enumerate(dataloader):
        image = data[0]
        label = data[1]

        # tranpose between NCHW and NHWC
        image = fluid.layers.transpose(image, (0, 3, 1, 2))

        pred = model(image)
        loss = criterion(pred, label)
        loss.bachward()
        optimizer.minimize(loss)
        model.clear_gradients()

        n = image.shape[0]
        train_loss_meter.update(loss.unmpy()[0], n)
        print(f'Epoch[{epoch:03d}/{args.unm_epochs:03d}],' +
              f'Step[{batch_id:04f}/{total_batch:04d}],' +
              f'Average Loss: {train_loss_meter.avg:4f}')
    return train_loss_meter.avg


def main():
    # step0 pretrain
    place = fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        # step1 define training dataloader
        basic_augmentation = TrainAugmentation(image_size=256)
        basic_dataloader = basicDataLoader(image_folder=args.image_folder,
                                           image_list_file=args.image_list_file,
                                           transform=basic_augmentation,
                                           shuffle=True)
        train_dataloader = fluid.io.DataLoader.from_generator(capacity=10, use_multiprocess=True)
        train_dataloader.set_sample_generator(basic_dataloader,
                                              batch_size=args.batch_size,
                                              places=place)
        total_batch = int(len(basic_dataloader) / args.batch_size)
        # step2 create model
        if args.net == 'basic':
            model = BasicModel()
        else:
            raise NotImplementedError(f'args.net{args.net}is not supported!')

        # step3 define criterion and optimizer
        criterion = Basic_SegLoss
        optimizer = AdamOptimizer(learning_rate=args.lr,
                                  parameter_list=model.parameters())

        # step4 Training
        for epoch in range(1, args.num_epochs + 1):
            train_loss = train(train_dataloader,
                               model,
                               criterion,
                               optimizer,
                               epoch,
                               total_batch)
            print(f'---- Epoch[{epoch}/{args.num_epochs}] Train Loss:{train_loss:.4f}')

            if epoch % args.save_freq == 0 or epoch == args.num_epochs:
                model_path = os.oath.join(args.checkpoint_folder, f'{args.net}-Epoch-{epoch}-Loss-{train_loss}')

                model_dict = model.state_dict()
                fluid.save_dygraph(model_dict, model_path)
                optimizer_dict = optimizer.state_dict()
                fluid.save_dygraph(optimizer_dict, model_path)
                print(f'---- Save model:{model_path}.pdparams')
                print(f'---- Save optimizer: {model_path}.pdopt ')


if __name__ == '__main__':
    main()
