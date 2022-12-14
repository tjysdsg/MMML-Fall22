import argparse
import ruamel_yaml as yaml
from pathlib import Path
import torch
from data import create_dataset, create_loader
from data.webqa_dataset import webqa_collate_fn
from models.blip_webqa import blip_vqa
import torch.nn.functional as F


@torch.no_grad()
def main(args, config):
    # dataset
    dataset, _, _ = create_dataset(
        dict(
            image_size=480,
            train_file=r'E:\webqa\data\WebQA_train_val.json',
            val_file=r'E:\webqa\data\WebQA_train_val.json',
            test_file=r'E:\webqa\data\WebQA_train_val.json',
            image_dir=r'E:\webqa\data\images',
        ),
        use_num_samples=100,
    )
    train_loader = create_loader(
        [dataset], [None],
        batch_size=[2],
        num_workers=[1], is_trains=[True],
        collate_fns=[webqa_collate_fn]
    )[0]

    # model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = blip_vqa(
        pretrained=r'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_14M.pth',
        image_size=config['image_size'],
        vit=config['vit'],
        vit_grad_ckpt=config['vit_grad_ckpt'],
        vit_ckpt_layer=config['vit_ckpt_layer'],
        multitask_qcate=True
    )
    model = model.to(device)

    image = torch.zeros((3, 2, 3, 480, 480)).to(device)
    questions = [
        'why are you running',
        'can you speak english',
        'your laptop is a junk',
    ]
    captions = [
        ['hey you', 'shut up dude'],
        ['screw you', 'i do not know what you mean', 'what are you doing'],
        ['this homework is killing me'],
    ]
    answers = [
        'fuck you',
        'screw you',
        'motherfucker shut your fucking mouth',
    ]
    n_img_facts = [
        2,
        0,
        1
    ]
    loss, _, _ = model(image, captions, questions, answers, n_img_facts, train=True)

    for images, captions, question, answer, n_img_facts, question_ids, qcates, retr_labels in train_loader:
        print('QUESTION:', question)
        print('CAPTIONS:', captions)
        print('ANSWER:', answer)

        # visualize images
        # batch_size, nf, channel, H, W = images.shape

        # # print(f'(batch_size, n_img_facts, channel, H, W):', images.shape)
        # for b in range(batch_size):
        #     for fi in range(nf):
        #         im = images[b, fi].detach().cpu().numpy()
        #         im = np.transpose(im, (1, 2, 0))

        #         plt.imshow(im)
        #         plt.savefig(
        #             os.path.join(
        #                 args.output_dir,
        #                 f'{question_ids[b]}_{fi}.jpg',
        #             )
        #         )
        #         plt.close('all')

        # run model
        images = images.to(device, non_blocking=True)
        (
            loss, mt_res, multimodal_cross_atts
        ) = model(images, captions, question, answer, n_img_facts, train=True)

        # MULTITASK
        mt_res = F.softmax(mt_res, dim=-1)
        print(torch.argmax(mt_res, dim=-1))

        # for ans, p, qid, qcate in zip(answer, pred, question_ids, qcates):
        #     print({"question_id": qid, 'qcate': qcate, "pred": p, "answer": ans})


def load_args_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/cased.yaml')
    parser.add_argument('--output_dir', default='output/WebQA')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    return args, config


if __name__ == '__main__':
    main(*load_args_configs())
