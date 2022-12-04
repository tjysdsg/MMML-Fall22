import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from data.webqa_dataset import WebQADataset
from transform.randaugment import RandomAugment


def create_dataset(config, min_scale=0.5, **kwargs) -> (WebQADataset, WebQADataset, WebQADataset):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(config['image_size'], scale=(min_scale, 1.0),
                                     interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 5, isPIL=True, augs=['Identity', 'AutoContrast', 'Brightness', 'Sharpness', 'Equalize',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size']), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = WebQADataset(config['train_file'], transform_train, config['image_dir'], split='train', **kwargs)
    val_dataset = WebQADataset(config['val_file'], transform_test, config['image_dir'], split='val', **kwargs)
    test_dataset = WebQADataset(config['test_file'], transform_test, config['image_dir'], split='test', **kwargs)
    return train_dataset, val_dataset, test_dataset


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
                                                      shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers,
                                                                    is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders


def test():
    datasets = create_dataset(
        dict(
            image_size=480,
            train_file=r'E:\webqa\data\WebQA_train_val.json',
            val_file=r'E:\webqa\data\WebQA_train_val.json',
            test_file=r'E:\webqa\data\WebQA_test.json',
            image_dir=r'E:\webqa\data\images',
        ),
        use_num_samples=100,
    )

    for data in datasets:
        print('=' * 80)
        print(data.split)
        print('=' * 80)
        (
            images,
            captions,
            Q,
            A,
            question_id,
            qcate,
            retrieval_labels,
        ) = data[50]
        print('captions:', captions)
        print('retr_labels:', retrieval_labels)

        from webqa_dataset import webqa_collate_fn
        loader = DataLoader(
            data,
            batch_size=4,
            num_workers=1,
            shuffle=True,
            collate_fn=webqa_collate_fn,
        )
        for (
                images, captions, question, answer, n_facts, question_ids, qcates, retr_labels,
        ) in loader:
            print('batch captions:', captions)
            print('batch retr_labels', retr_labels)
            break


if __name__ == '__main__':
    test()
