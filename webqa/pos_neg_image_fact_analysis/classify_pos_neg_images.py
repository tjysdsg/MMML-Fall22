from argparse import ArgumentParser
import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from torchvision.models import resnet50, ResNet50_Weights
from torch.optim.lr_scheduler import StepLR
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Accuracy, Precision, Recall, Loss
from ignite.handlers import Checkpoint, DiskSaver
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--train', type=str, default=r'E:\repos\MMML-Fall22\pos_neg_image_fact_analysis\train.tsv')
    parser.add_argument('--test', type=str, default=r'E:\repos\MMML-Fall22\pos_neg_image_fact_analysis\test.tsv')
    parser.add_argument('--val', type=str, default=r'E:\repos\MMML-Fall22\pos_neg_image_fact_analysis\val.tsv')
    parser.add_argument('--data-dir', type=str, default=r'E:\webqa\data\images',
                        help='Folder containing all image files')

    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--exp', type=str, default='exp')
    return parser.parse_args()


class PosNegImageClassifier(nn.Module):
    def __init__(self, model: nn.Module, num_classes=2):
        super().__init__()
        self.model = model

        self.linear = nn.Linear(self.model.fc.out_features, num_classes)
        self.input_size = 224

        # freeze resnet
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.model(x)
        return self.linear(x)


class ImageDataset(Dataset):
    def __init__(self, list_file: str, data_dir: str, data_transforms: transforms.Compose):
        super().__init__()
        self.data_dir = data_dir
        self.transforms = data_transforms

        self.data = []
        with open(list_file) as f:
            for line in f:
                qid, imgid, label = line.rstrip('\n').split()
                self.data.append([
                    os.path.join(self.data_dir, f'{int(imgid)}.jpg'),
                    int(label)
                ])

    def __getitem__(self, index):
        file, label = self.data[index]
        img = read_image(file, ImageReadMode.RGB)
        img = self.transforms(img)

        return img, label

    def __len__(self):
        return len(self.data)


def train(args):
    # https://pytorch.org/vision/stable/models.html
    # https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    # https://github.com/pytorch/ignite/blob/master/examples/notebooks/EfficientNet_Cifar100_finetuning.ipynb

    data_transforms = {  # similar to ResNet50_Weights.DEFAULT.transforms()
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Create training and validation datasets
    train_loader = DataLoader(
        ImageDataset(args.train, args.data_dir, data_transforms['train']),
        batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        ImageDataset(args.test, args.data_dir, data_transforms['test']),
        batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        ImageDataset(args.val, args.data_dir, data_transforms['test']),
        batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    # plot some training images
    # import torchvision.utils as vutils
    # from matplotlib import pyplot as plt
    # batch = next(iter(train_loader))
    # plt.figure(figsize=(16, 8))
    # plt.axis("off")
    # plt.title("Training Images")
    # plt.imshow(
    #     vutils.make_grid(batch[0][:16], padding=2, normalize=True).cpu().numpy().transpose((1, 2, 0))
    # )
    # plt.show()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PosNegImageClassifier(
        resnet50(weights=ResNet50_Weights.DEFAULT)
    ).to(device)

    optimizer = optim.SGD(model.linear.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001, nesterov=True)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.98)
    criterion = nn.CrossEntropyLoss()

    # run lr scheduler every epoch
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    metrics = {
        'Loss': Loss(criterion),
        'Accuracy': Accuracy(),
        'Precision': Precision(average=True),
        'Recall': Recall(average=True),
    }
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda engine: lr_scheduler.step())

    @trainer.on(Events.ITERATION_COMPLETED(every=10))
    def log_training_loss(trainer):
        print(f"Epoch{trainer.state.epoch} loss: {trainer.state.output:.2f}")

    evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_test_results(trainer):
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        print(f"Epoch{trainer.state.epoch} test_acc: {metrics['Accuracy']:.2f}")

    # store models
    disk_saver = DiskSaver(dirname='exp', require_empty=False)
    best_model_handler = Checkpoint(to_save={'model': model}, save_handler=disk_saver)
    evaluator.add_event_handler(Events.COMPLETED, best_model_handler)

    # setup tensorboard logger
    tb_logger = TensorboardLogger(log_dir="exp/tensorboard")
    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=10),
        tag="training",
        output_transform=lambda loss: {"batch_loss": loss},
    )
    tb_logger.attach_output_handler(
        evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag='testing',
        metric_names="all",
        global_step_transform=global_step_from_engine(trainer),
    )

    trainer.run(train_loader, max_epochs=5)
    tb_logger.close()


def main():
    args = get_args()
    train(args)


if __name__ == '__main__':
    main()
