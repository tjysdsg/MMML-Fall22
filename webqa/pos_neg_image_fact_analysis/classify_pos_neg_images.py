import json
from argparse import ArgumentParser
import os
from typing import Literal
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from torchvision.models import resnet50, ResNet50_Weights
from torch.optim.lr_scheduler import StepLR
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer, Engine
from ignite.metrics import Accuracy, Precision, Recall, Loss
from ignite.handlers import Checkpoint, DiskSaver
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--data-json', type=str, default=r'E:\webqa\data\WebQA_train_val.json',
                        help='Path to the data json file')
    parser.add_argument('--train', type=str,
                        default=r'E:\repos\MMML-Fall22\webqa\pos_neg_image_fact_analysis\train.tsv')
    parser.add_argument('--test', type=str, default=r'E:\repos\MMML-Fall22\webqa\pos_neg_image_fact_analysis\test.tsv')
    parser.add_argument('--val', type=str, default=r'E:\repos\MMML-Fall22\webqa\pos_neg_image_fact_analysis\val.tsv')
    parser.add_argument('--data-dir', type=str, default=r'E:\webqa\data\images',
                        help='Folder containing all image files')

    parser.add_argument('--pred-type', type=str, default='pos_neg', choices=['pos_neg', 'topics', 'qcate'])
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--exp', type=str, default='exp')
    return parser.parse_args()


class PosNegImageClassifier(nn.Module):
    def __init__(self, model: nn.Module, num_classes: int):
        super().__init__()
        self.model = model

        self.linear = nn.Linear(self.model.fc.out_features, num_classes)
        self.input_size = 224

        # freeze resnet
        # for param in self.model.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        x = self.model(x)
        return self.linear(x)


class ImageDataset(Dataset):
    def __init__(self, data_json: str, list_file: str, data_dir: str, data_transforms: transforms.Compose,
                 pred_type: Literal['pos_neg', 'topics', 'qcate']):
        super().__init__()
        with open(data_json) as f:
            self.meta = json.load(f)

        self.data_dir = data_dir
        self.transforms = data_transforms

        # map qcate to index
        self.qcate2index = {
            'choose': 0, 'number': 1, 'shape': 2, 'YesNo': 3, 'color': 4, 'Others': 5,
        }

        # map topics to indices
        all_topics = [
            'Olympic stadium',
            'Olympic game venue',
            'olympic torch',
            'wind instruments',
            'instruments',
            'Indianapolis Motor Speedway',
            'civic center',
            'public art',
            'mushroom',
            'carnivores',
            'the day of the dead',
            'strange architecture',
            'Soho',
            'rodent',
            'college libraries',
            'cacti',
            'vista',
            'space craft',
            'french famous paintings',
            'Neo-Impressionism art',
            'millitary parades',
            'olympics athletics track and field',
            'olympic village',
            'beetle',
            'dining',
            'tourist attractions',
            'modern artwork',
            'butterfly',
            'art college buildings',
            'museum',
            'keyboard instruments',
            'U.S. Coins',
            'indigenous American',
            'insect',
            'civil war memorial',
            'string instruments',
            'mural',
            'flora',
            "world's best goalkeeper",
            'festival',
            'ethnic clothing',
            'streets',
            'winter olympics',
            'organ',
            'plants',
            'deer',
            'car',
            'NBA basketball match',
            'drum',
            'research stations antarctica',
            'Neoclassicism art',
            'hall of fame',
            'Olympic athletics equipment',
            'jellyfish',
            'fish',
            'tech institute',
            'frog',
            'downtown',
            'mall',
            'Extreme Sports',
            'plaza',
            'french museums',
            'space station',
            'Unique Skyscrapers',
            'olympics opening ceremony',
            'world expo pavilion',
            'youth olympics',
            'artists',
            'renaissance art paintings',
            'Other',
            'bird',
            'public art general',
            'monkey',
            'Christ Church Cathedral',
        ]
        self.topic2index = {s: i for i, s in enumerate(all_topics)}

        self.data = []
        with open(list_file) as f:
            for line in f:
                qid, imgid, pos_neg = line.rstrip('\n').split()

                # topics
                topic = self.topic2index[self.meta[qid]['topic']]

                # qcate
                qcate = self.qcate2index[self.meta[qid]['Qcate']]

                self.data.append([
                    os.path.join(self.data_dir, f'{int(imgid)}.jpg'),
                    int(pos_neg),
                    topic,
                    qcate,
                ])

        # determine classifier output type
        if pred_type == 'pos_neg':
            self.y_idx = 1
            self.num_classes = 2
        elif pred_type == 'topics':
            self.y_idx = 2
            self.num_classes = len(self.topic2index)
        elif pred_type == 'qcate':
            self.y_idx = 3
            self.num_classes = len(self.qcate2index)
        else:
            raise RuntimeError(f"Invalid --pred-type={pred_type}")

    def __getitem__(self, index):
        file = self.data[index][0]
        img = read_image(file, ImageReadMode.RGB)
        img = self.transforms(img)

        label = self.data[index][self.y_idx]
        return img, label

    def __len__(self):
        return len(self.data)

    @property
    def out_classes(self):
        return self.num_classes


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
    train_set = ImageDataset(args.data_json, args.train, args.data_dir, data_transforms['train'], args.pred_type)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        ImageDataset(args.data_json, args.test, args.data_dir, data_transforms['test'], args.pred_type),
        batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        ImageDataset(args.data_json, args.val, args.data_dir, data_transforms['test'], args.pred_type),
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
        resnet50(weights=ResNet50_Weights.DEFAULT),
        num_classes=train_set.out_classes,
    ).to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001, nesterov=True)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.98)
    criterion = nn.CrossEntropyLoss()

    # create trainer
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    metrics = {
        'Loss': Loss(criterion),
        'Accuracy': Accuracy(),
        'Precision': Precision(average=True),
        'Recall': Recall(average=True),
    }

    # run lr scheduler every epoch
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda engine: lr_scheduler.step())

    @trainer.on(Events.ITERATION_COMPLETED(every=10))
    def log_training_loss(trainer):
        print(f"Epoch{trainer.state.epoch} loss: {trainer.state.output:.2f}")

    # evaluator for validation
    evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_test_results(trainer):
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        print(f"Epoch{trainer.state.epoch} test_acc: {metrics['Accuracy']:.2f}")

    # store models
    disk_saver = DiskSaver(dirname='exp', require_empty=False)
    to_save = {'trainer': trainer, 'model': model, 'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
    best_model_handler = Checkpoint(to_save=to_save, save_handler=disk_saver)
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

    trainer.run(train_loader, max_epochs=100)
    tb_logger.close()


def main():
    args = get_args()
    train(args)


if __name__ == '__main__':
    main()
