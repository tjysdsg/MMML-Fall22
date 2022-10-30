from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
from logger import get_logger
import os
import glob
import math
import json
import argparse
from tqdm import tqdm, trange
from pathlib import Path
import numpy as np
import torch
import torch.multiprocessing as mp

mp.set_start_method('spawn', force=True)
from torch.utils.data import DataLoader
import random
import copy
from pytorch_pretrained_bert.modeling import BertForWebqa
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from vlp.loader_utils import batch_list_to_batch_tensors
from vlp.webqa_dataset import WebQARetrievalDataset
import vlp.webqa_loader as webqa_loader
import matplotlib.pyplot as plt
from typing import List


def _get_max_epoch_model(output_dir):
    fn_model_list = glob.glob(os.path.join(output_dir, "model.*.bin"))
    fn_optim_list = glob.glob(os.path.join(output_dir, "optim.*.bin"))
    if (not fn_model_list) or (not fn_optim_list):
        return None
    both_set = set([int(Path(fn).stem.split('.')[-1]) for fn in fn_model_list]
                   ) & set([int(Path(fn).stem.split('.')[-1]) for fn in fn_optim_list])
    if both_set:
        return max(both_set)
    else:
        return None


def _get_loader_from_dataset(train_dataset, train_batch_size, num_workers, collate_fn):
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    return train_dataloader


def save_loss_curve(loss, i_epoch, output_dir, task, all_tasks):
    plt.figure(figsize=(12, 5))
    plt.plot(range(1, len(loss) + 1), loss)
    plt.xlabel("iter")
    plt.ylabel("loss")
    title = "{}__epc={}__all_tasks={}".format(task, i_epoch, all_tasks)
    plt.title(title)
    plt.savefig(os.path.join(output_dir, "figs/" + title + ".jpg"))


def get_dataloaders(args, device):
    from pytorch_pretrained_bert.tokenization import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case, cache_dir=args.output_dir + '/.pretrained_model'
    )
    if args.max_position_embeddings:
        tokenizer.max_len = args.max_position_embeddings

    processor = webqa_loader.WebQADataSampleProcessor(
        args.max_pred, args.mask_prob,
        list(tokenizer.vocab.keys()), tokenizer.convert_tokens_to_ids,
        seed=args.seed, max_len=args.max_seq_length,
        len_vis_input=args.len_vis_input, max_len_a=args.max_len_a,
        max_len_b=args.max_len_b,
        max_len_img_cxt=args.max_len_img_cxt,
        new_segment_ids=args.new_segment_ids,
        truncate_config={
            'trunc_seg': args.trunc_seg,
            'always_truncate_tail': args.always_truncate_tail
        },
        use_img_meta=args.use_img_meta,
        use_img_content=args.use_img_content,
        use_txt_fact=args.use_txt_fact
    )

    train_dataloaders = []
    if "filter" in args.task_to_learn:
        if "txt" in args.answer_provided_by:
            train_dataset = WebQARetrievalDataset(
                dataset_json_path=args.txt_dataset_json_path,
                split=args.split,
                Qcate=args.Qcate,
                batch_size=args.train_batch_size,
                tokenizer=tokenizer,
                feature_folder=args.feature_folder,
                use_num_samples=args.use_num_samples,
                processor=processor,
                answer_provided_by='txt',
                use_x_distractors=args.use_x_distractors,
                max_snippets=args.txt_filter_max_choices,
                max_imgs=args.img_filter_max_choices,
                imgid_map=args.image_id_map_path,
                device=device,
            )
            train_dataloader = _get_loader_from_dataset(
                train_dataset,
                args.train_batch_size,
                args.num_workers,
                batch_list_to_batch_tensors,
            )
            train_dataloaders.append(train_dataloader)

        if "img" in args.answer_provided_by:
            train_dataset = WebQARetrievalDataset(
                dataset_json_path=args.img_dataset_json_path,
                split=args.split,
                Qcate=args.Qcate,
                batch_size=args.train_batch_size,
                tokenizer=tokenizer,
                feature_folder=args.feature_folder,
                use_num_samples=args.use_num_samples,
                processor=processor,
                answer_provided_by='img',
                use_x_distractors=args.use_x_distractors,
                max_snippets=args.txt_filter_max_choices,
                max_imgs=args.img_filter_max_choices,
                imgid_map=args.image_id_map_path,
                device=device,
            )
            train_dataloader = _get_loader_from_dataset(
                train_dataset,
                args.train_batch_size, args.num_workers,
                batch_list_to_batch_tensors,
            )
            train_dataloaders.append(train_dataloader)

    if "qa" in args.task_to_learn:
        if "txt" in args.answer_provided_by:
            train_dataset = webqa_loader.webqaDataset_qa(dataset_json_path=args.txt_dataset_json_path, split=args.split,
                                                         Qcate=args.Qcate,
                                                         batch_size=args.train_batch_size, tokenizer=tokenizer,
                                                         use_num_samples=args.use_num_samples,
                                                         processor=processor, device=device)

            train_dataloader = _get_loader_from_dataset(train_dataset,
                                                        args.train_batch_size, args.num_workers,
                                                        batch_list_to_batch_tensors)
            train_dataloaders.append(train_dataloader)

        if "img" in args.answer_provided_by:
            train_dataset = webqa_loader.webqaDataset_qa_with_img(dataset_json_path=args.img_dataset_json_path,
                                                                  split=args.split, Qcate=args.Qcate,
                                                                  batch_size=args.train_batch_size, tokenizer=tokenizer,
                                                                  feature_folder=args.feature_folder,
                                                                  use_num_samples=args.use_num_samples,
                                                                  processor=processor, imgid_map=args.image_id_map_path,
                                                                  device=device)
            train_dataloader = _get_loader_from_dataset(train_dataset,
                                                        args.train_batch_size, args.num_workers,
                                                        batch_list_to_batch_tensors)
            train_dataloaders.append(train_dataloader)

    return train_dataloaders


def model_forward_pass(model, batch, device, n_gpu, drop_worst_ratio=0):
    batch = [t.to(device) if isinstance(t, torch.Tensor) else t for t in batch]
    (
        input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next, do_filter_task,
        filter_label, logit_mask, ori_choices, task_idx, img, vis_pe, context, cxt_modality_label,
        example_ids
    ) = batch

    # Each batch contains img/text facts + question (+answer)

    # Input of filter task:
    # batch_size 1 float16
    # sample: 1 fact question answer

    #       Image: [CLS], [[RCNN feats] [Image captions] of each fact], [SEP], [Question] [Answer] [SEP]
    #       Text: [CLS], [[Text fact], [SEP] of each fact], [Question] [Answer] [SEP]
    # Input of QA task:
    #       Image: [CLS], [[RCNN feats] [Image captions] of each fact], [SEP], [Question] [SEP]
    #       Text: [CLS], [[Text fact] of each fact], [SEP], [Question] [SEP]
    conv_feats = img.data  # B x 100 bounding boxes x 2048
    vis_pe = vis_pe.data  # positional embeddings

    # input_ids contains the actual input sequence,
    # but the tokens of image and text facts are marked as [UNK] at this stage,
    # their values will be set to the sequence in BertEmbedding forward call

    loss_tuple = model(
        vis_feats=conv_feats, vis_pe=vis_pe, input_ids=input_ids, token_type_ids=segment_ids,
        attention_mask=input_mask,
        masked_lm_labels=masked_ids, do_filter_task=do_filter_task,
        filter_label=filter_label, logit_mask=logit_mask, context=context,
        cxt_modality_label=cxt_modality_label, next_sentence_label=is_next,
        masked_pos=masked_pos, masked_weights=masked_weights,
        task_idx=task_idx,
        drop_worst_ratio=drop_worst_ratio
    )
    mean_reward = loss_tuple[0].new(1).fill_(0)

    # disable pretext_loss_deprecated for now
    masked_lm_loss, cls_loss = loss_tuple
    if n_gpu > 1:  # mean() to average on multi-gpu. For dist, this is done through gradient addition.
        masked_lm_loss = masked_lm_loss.mean()
        cls_loss = cls_loss.mean()
    loss = masked_lm_loss + cls_loss

    return loss, masked_lm_loss, cls_loss, mean_reward


def train(
        logger: logging.Logger, args,
        model: BertForWebqa, device, optimizer, n_gpu: int,
        recover_step: int, global_step: int, t_total: int,
        train_dataloaders: List[DataLoader], train_dataloader_order: List[int],
):
    if args.amp:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2")

    logger.info("========================\nStart training\n========================\n")

    if "img" in args.answer_provided_by:
        logger.info(f"use_img_meta = {args.use_img_meta}")
        logger.info(f"use_img_content = {args.use_img_content}")
        logger.info(f"\nimg Filter_max_choices: {args.img_filter_max_choices}")
        if args.use_x_distractors:
            logger.info(f"\ntxt Filter_max_choices: {args.txt_filter_max_choices}")
    if "txt" in args.answer_provided_by:
        logger.info("use_txt_fact = ", args.use_txt_fact)
        logger.info(f"\ntxt Filter_max_choices: {args.txt_filter_max_choices}")
        if args.use_x_distractors:
            logger.info(f"\nimg Filter_max_choices: {args.img_filter_max_choices}")

    logger.info("***** Running training *****")
    logger.info(f"  Batch size = {args.train_batch_size}")
    logger.info(f"  Num steps = {t_total}")

    model.train()
    if recover_step:
        start_epoch = recover_step + 1
    else:
        start_epoch = 1
    for i_epoch in trange(start_epoch, args.num_train_epochs + 1, desc="Epoch"):
        dataloader_iters = [iter(l) for l in train_dataloaders]
        iter_bar = tqdm(train_dataloader_order, desc='Iter (loss=X.XXX), loader_idx=X')

        qa_loss = []
        filter_loss = []
        loss_dict = [[], [], [], []]
        scst_reward = []
        for step, loader_idx in enumerate(iter_bar):
            batch = next(dataloader_iters[loader_idx])
            loss, masked_lm_loss, cls_loss, mean_reward = model_forward_pass(
                model, batch, device, n_gpu,
                drop_worst_ratio=args.max_drop_worst_ratio if i_epoch > args.drop_after else 0
            )

            # logging for each step (i.e., before normalization by args.gradient_accumulation_steps)
            iter_bar.set_description('Iter (loss={:.3f}) loader_idx={}'.format(loss.item(), loader_idx))
            qa_loss.append(masked_lm_loss.item())
            filter_loss.append(cls_loss.item())
            loss_dict[loader_idx].append(loss.item())
            scst_reward.append(mean_reward.item())
            # logger.info("\n ---------------------- loss.grad ------------------------ \n")
            # for name, parms in model.named_parameters():
            # logger.info('-->name:', name, '-->grad_requirs:',parms.requires_grad, ' -->grad_value:',parms.grad)

            if step % 100 == 0:
                logger.info(
                    f"Epoch {i_epoch}, Iter {step}, Loss {np.mean(qa_loss):.2f}, "
                    f"Filter {np.mean(filter_loss):.2f}, Mean R {np.mean(scst_reward):.3f}\n"
                )

            # ensure that accumulated gradients are normalized
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.amp:
                from apex import amp
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
                if args.amp:
                    # modify learning rate with special warm up BERT uses
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        logger.info(qa_loss)
        logger.info(filter_loss)
        logger.info(loss_dict)

        # Save a trained model
        logger.info("** ** * Saving fine-tuned model and optimizer ** ** * ")
        model_to_save = model.module if hasattr(
            model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(
            args.ckpts_dir, "model.{0}.bin".format(i_epoch))
        output_optim_file = os.path.join(
            args.ckpts_dir, "optim.{0}.bin".format(i_epoch))

        torch.save(copy.deepcopy(model_to_save).cpu().state_dict(), output_model_file)
        torch.save(optimizer.state_dict(), output_optim_file)

        # Save loss curve
        if args.save_loss_curve:
            loss_idx = 0
            if "filter" in args.task_to_learn and "txt" in args.answer_provided_by:
                save_loss_curve(loss_dict[loss_idx], i_epoch, args.output_dir, "filter-txt",
                                "-".join([args.task_to_learn, args.answer_provided_by]))
                loss_idx += 1
            if "filter" in args.task_to_learn and "img" in args.answer_provided_by:
                save_loss_curve(loss_dict[loss_idx], i_epoch, args.output_dir, "filter-img",
                                "-".join([args.task_to_learn, args.answer_provided_by]))
                loss_idx += 1
            if "qa" in args.task_to_learn and "txt" in args.answer_provided_by:
                save_loss_curve(loss_dict[loss_idx], i_epoch, args.output_dir, "qa-txt",
                                "-".join([args.task_to_learn, args.answer_provided_by]))
                loss_idx += 1
            if "qa" in args.task_to_learn and "img" in args.answer_provided_by:
                save_loss_curve(loss_dict[loss_idx], i_epoch, args.output_dir, "qa-img",
                                "-".join([args.task_to_learn, args.answer_provided_by]))
                loss_idx += 1
        logger.info("***** CUDA.empty_cache() *****")
        torch.cuda.empty_cache()


def main():
    args = get_args()

    args.max_seq_length = args.max_len_b + args.max_len_a + 3  # +3 for 2x[SEP] and [CLS]
    args.use_img_meta = not args.no_img_meta
    args.use_img_content = not args.no_img_content
    args.use_txt_fact = not args.no_txt_fact
    assert args.len_vis_input == 100, "run main: only support 100 region features per image"

    # output config
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.ckpts_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "figs"), exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, args.log_file), __file__)

    # dump running args
    opt_json = os.path.join(args.output_dir, 'opt.json')
    json.dump(args.__dict__, open(opt_json, 'w'), sort_keys=True, indent=2)
    logger.info(f"Commandline args are saved in {opt_json}")

    # determine cpu or gpu
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info(f"Device: {device} n_gpu: {n_gpu}, amp: {args.amp}")

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            f"Invalid gradient_accumulation_steps parameter: {args.gradient_accumulation_steps}, should be >= 1"
        )

    # fix random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # prepare dataloaders
    train_dataloaders = get_dataloaders(args, device)
    loader_lengths = [len(e) for e in train_dataloaders]
    logger.info(f"n_batches = {sum(loader_lengths)}", )

    # train_dataloader_order randomly specifies which at each iteration which dataloader to retrieve a batch from,
    # in case multiple dataloaders (text+image) are used during training
    # If there is only one dataloader, the training loop is equivalent to `for i, batch in enumerate(dataloder):`
    train_dataloader_order = []
    for i in range(len(loader_lengths)):
        train_dataloader_order.extend([i] * loader_lengths[i])
    random.shuffle(train_dataloader_order)
    logger.info(f"\ntrain_dataloader_order = {train_dataloader_order}")

    # Total number of times we update the model's parameter, note we are using grad accum
    t_total = int(sum(loader_lengths) * args.num_train_epochs * 1. / args.gradient_accumulation_steps)

    # Prepare model
    recover_step = _get_max_epoch_model(args.ckpts_dir)
    if args.recover_step: recover_step = args.recover_step
    if args.recover_ori_ckpt or args.from_scratch: recover_step = None
    if args.from_scratch: args.model_recover_path = None

    cls_num_labels = 2
    type_vocab_size = 6 if args.new_segment_ids else 2
    relax_projection = 4 if args.relax_projection else 0
    task_idx_proj = 3  # harded to be 3 # if args.tasks == 'img2txt' else 0

    # index in BERT vocab: 103, 102, 0
    # mask_word_id, eos_word_ids, pad_word_ids = tokenizer.convert_tokens_to_ids(["[MASK]", "[SEP]", "[PAD]"])

    # check if detectron weights and bias files are present
    fc7_weight_path = os.path.join(args.detectron_dir, 'fc7_w.pkl')
    fc7_bias_path = os.path.join(args.detectron_dir, 'fc7_b.pkl')
    if not os.path.exists(fc7_weight_path) or not os.path.exists(fc7_bias_path):
        raise FileNotFoundError(f"Cannot find fc7_b.pkl and fc7_w.pkl under {args.detectron_dir}")

    # Load model checkpoint
    if recover_step is None and args.model_recover_path is None:
        logger.info("----------------------- Nothing to recover -------------------------")
        _state_dict = {} if args.from_scratch else None
        model = BertForWebqa.from_pretrained(
            args.bert_model, state_dict=_state_dict, num_labels=cls_num_labels,
            type_vocab_size=type_vocab_size, relax_projection=relax_projection,
            config_path=args.config_path, task_idx=task_idx_proj,
            max_position_embeddings=args.max_position_embeddings, label_smoothing=args.label_smoothing,
            cache_dir=args.output_dir + '/.pretrained_model',
            drop_prob=args.drop_prob, max_len_img_cxt=args.max_len_img_cxt,
            fc7_weight_path=fc7_weight_path, fc7_bias_path=fc7_bias_path,
        )
        global_step = 0
    else:
        if recover_step:
            logger.info(f"***** Recover model from step {recover_step} *****")
            model_recover = torch.load(os.path.join(args.ckpts_dir, "model.{0}.bin".format(recover_step)))
            # recover_step == number of epochs
            global_step = math.floor(recover_step * t_total * 1. / args.num_train_epochs)
        else:  # elif args.model_recover_path:
            logger.info(f"***** Recover model from path: {args.model_recover_path} *****")
            model_recover = torch.load(args.model_recover_path)
            global_step = 0

        model = BertForWebqa.from_pretrained(
            args.bert_model, state_dict=model_recover, num_labels=cls_num_labels,
            type_vocab_size=type_vocab_size, relax_projection=relax_projection,
            config_path=args.config_path, task_idx=task_idx_proj,
            max_position_embeddings=args.max_position_embeddings, label_smoothing=args.label_smoothing,
            cache_dir=args.output_dir + '/.pretrained_model',
            drop_prob=args.drop_prob, max_len_img_cxt=args.max_len_img_cxt,
            fc7_weight_path=fc7_weight_path, fc7_bias_path=fc7_bias_path,
        )

        del model_recover
        torch.cuda.empty_cache()

    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=[1, 0])
        logger.info(f"\nn_gpu = {n_gpu}")

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.amp:
        try:
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use mixed precision training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False)  # FIXME: max_grad_norm=1.0
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        # FIXME: use BertAdam?
        # optimizer = BertAdam(optimizer_grouped_parameters,
        # lr=args.learning_rate,
        # warmup=args.warmup_proportion,
        # schedule=args.sche_mode,
        # t_total=t_total,
        # weight_decay = args.weight_decay)

    if recover_step and args.do_train:
        logger.info(f"***** Recover optimizer: {recover_step} *****")
        optim_recover = torch.load(os.path.join(
            args.ckpts_dir, "optim.{0}.bin".format(recover_step)))
        if hasattr(optim_recover, 'state_dict'):
            optim_recover = optim_recover.state_dict()
        optimizer.load_state_dict(optim_recover)

    logger.info("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()

    if args.do_train:  # run training
        train(
            logger, args, model, device, optimizer, n_gpu, recover_step, global_step, t_total,
            train_dataloaders, train_dataloader_order
        )

    elif args.do_val:  # run validation on data
        logger.info("--------------- Compute loss without grad ------------------")
        assert args.split == "val"
        if "img" in args.answer_provided_by:
            logger.info(f"use_img_meta = {args.use_img_meta}", args.use_img_meta)
            logger.info(f"use_img_content = {args.use_img_content}")
            logger.info(f"\nimg Filter_max_choices: {args.img_filter_max_choices}")
        if "txt" in args.answer_provided_by:
            logger.info(f"use_txt_fact = ", args.use_txt_fact)
            logger.info(f"\ntxt Filter_max_choices: {args.txt_filter_max_choices}")

        logger.info("***** Compute loss without grad *****")
        logger.info(f"  Batch size = {args.train_batch_size}")
        logger.info(f"  Num steps = {t_total}")

        model.eval()

        with torch.no_grad():
            dataloader_iters = [iter(l) for l in train_dataloaders]
            iter_bar = tqdm(train_dataloader_order, desc='Iter (loss=X.XXX), loader_idx=X')

            qa_loss = []
            filter_loss = []
            loss_dict = [[], [], [], []]
            scst_reward = []
            for step, loader_idx in enumerate(iter_bar):
                batch = next(dataloader_iters[loader_idx])
                loss, masked_lm_loss, cls_loss, mean_reward = model_forward_pass(
                    model, batch, device, n_gpu, drop_worst_ratio=0
                )

                # logging for each step (i.e., before normalization by args.gradient_accumulation_steps)
                iter_bar.set_description('Iter (loss={:.3f}) loader_idx={}'.format(loss.item(), loader_idx))
                qa_loss.append(masked_lm_loss.item())
                filter_loss.append(cls_loss.item())
                loss_dict[loader_idx].append(loss.item())
                scst_reward.append(mean_reward.item())

                if step % 100 == 0:
                    logger.info(
                        f"Iter {step}, Loss {np.mean(qa_loss):.2f}, Filter {np.mean(filter_loss):.2f}, "
                        f"Mean R {np.mean(scst_reward):.3f}\n"
                    )

            logger.info(qa_loss)
            logger.info(filter_loss)
            logger.info(loss_dict)
            logger.info(f"Mean loss = {np.mean([l for L in loss_dict for l in L])}")

            torch.cuda.empty_cache()

            with open(os.path.join(args.output_dir, "val_loss.txt"), "a") as f:
                f.write("\nrecover_step = {}, use_num_samples = {}, answer_provided_by = {}, task = {}\n".format(
                    recover_step, args.use_num_samples, args.answer_provided_by, args.task_to_learn))
                f.write(str(np.mean([l for L in loss_dict for l in L])))

    elif args.do_predict:  # inference mode
        if "img" in args.answer_provided_by:
            logger.info(f"use_img_content = {args.use_img_content}")
            logger.info(f"use_img_meta = {args.use_img_meta}")
            logger.info(f"\nimg Filter_max_choices: {args.img_filter_max_choices}")
            if args.use_x_distractors:
                logger.info(f"\ntxt Filter_max_choices: {args.txt_filter_max_choices}")

        if "txt" in args.answer_provided_by:
            logger.info(f"use_txt_fact = {args.use_txt_fact}")
            logger.info(f"\ntxt Filter_max_choices: {args.txt_filter_max_choices}")
            if args.use_x_distractors:
                logger.info(f"\nimg Filter_max_choices: {args.img_filter_max_choices}")

        logger.info("-------------------- Filter Inference mode ------------------------")
        logger.info(f"split = {args.split}")
        logger.info(f"use_num_samples = {args.use_num_samples}")

        th_list = [float(i) for i in args.filter_infr_th.split("|")]
        if not 0.7 in th_list: th_list.append(0.7)
        logger.info(f"\nThresholds: {str(th_list)}")
        model.eval()

        score_dict = dict([(th, {'pr': [], 're': [], 'f1': []}) for th in th_list])
        # for th in th_list:
        dataloader_iters = [iter(l) for l in train_dataloaders]
        total_samples = sum([len(l.dataset) for l in train_dataloaders])
        logger.info(f"\ntotal_samples = {total_samples}")
        iter_bar = tqdm(train_dataloader_order, desc='Iter (loss=X.XXX), loader_idx=X')

        Pred = []
        Choices = []
        Example_ids = []
        Filter_labels = []
        with torch.no_grad():
            for step, loader_idx in enumerate(iter_bar):
                batch = next(dataloader_iters[loader_idx])
                batch = [t.to(device) if not isinstance(t, list) else t for t in batch]
                input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next, do_filter_task, filter_label, logit_mask, ori_choices, task_idx, img, vis_pe, context, cxt_modality_label, example_ids = batch

                conv_feats = img.data  # Bx100x2048
                vis_pe = vis_pe.data
                cur_batch_score, pred = model(vis_feats=conv_feats, vis_pe=vis_pe, input_ids=input_ids,
                                              token_type_ids=segment_ids, attention_mask=input_mask,
                                              masked_lm_labels=masked_ids, do_filter_task=do_filter_task,
                                              filter_label=filter_label, logit_mask=logit_mask, context=context,
                                              cxt_modality_label=cxt_modality_label, next_sentence_label=is_next,
                                              masked_pos=masked_pos, masked_weights=masked_weights,
                                              task_idx=task_idx, drop_worst_ratio=0, filter_infr_th=th_list)
                assert len(cur_batch_score) == len(th_list)
                Pred.append(pred)
                Filter_labels.append(filter_label.detach().cpu())
                Choices.extend(ori_choices)
                Example_ids.extend(example_ids)
                if "filter" in args.task_to_learn:
                    for th in cur_batch_score:
                        score_dict[th]['pr'].append(cur_batch_score[th][0])
                        score_dict[th]['re'].append(cur_batch_score[th][1])
                        score_dict[th]['f1'].append(cur_batch_score[th][2])
                    iter_bar.set_description('Iter={} loader_idx={} '.format(step, loader_idx))
                else:
                    raise ValueError("Currently don't support qa task in inference mode")

            Pred = torch.cat(Pred, dim=0)
            Pred = Pred.numpy()
            Pred = [["{0:.4f}".format(s) for s in p] for p in Pred]
            Filter_labels = torch.cat(Filter_labels, dim=0)
            Filter_labels = Filter_labels.numpy()
            Filter_labels = [[int(i[0]) for i in b] for b in Filter_labels]
            for th in th_list:
                score_dict[th]['pr'] = np.sum(score_dict[th]['pr']) / float(total_samples)
                score_dict[th]['re'] = np.sum(score_dict[th]['re']) / float(total_samples)
                score_dict[th]['f1'] = np.sum(score_dict[th]['f1']) / float(total_samples)

                logger.info(f"\nth = {th}")
                logger.info(f"pr.mean = {score_dict[th]['pr']}")
                logger.info(f"re.mean = {score_dict[th]['re']}")
                logger.info(f"f1.mean = {score_dict[th]['f1']}")
        output_pkl = {}
        for e, c, l, p in zip(Example_ids, Choices, Filter_labels, Pred):
            output_pkl[e] = {"choices": c, "labels": l, "pred_scores": p}
        pkl_filename = "{}_{}_step{}".format(str(args.split), args.use_num_samples, recover_step)
        if "img" in args.answer_provided_by:
            args.output_suffix = args.img_dataset_json_path.split('/')[-1].replace(".json", "") + args.output_suffix
            pkl_filename += "_{}_{}_{}_{}_".format("img", args.img_filter_max_choices, args.use_img_content,
                                                   args.use_img_meta)
        if "txt" in args.answer_provided_by:
            args.output_suffix = args.txt_dataset_json_path.split('/')[-1].replace(".json", "") + args.output_suffix
            pkl_filename += "_{}_{}_{}_".format("txt", args.txt_filter_max_choices, args.use_txt_fact)
        pkl_filename += args.output_suffix
        if args.use_x_distractors: pkl_filename += "_UNknown_modality"

        with open(os.path.join(args.output_dir, "{}.json".format(pkl_filename)), "w") as f:
            json.dump(output_pkl, f, indent=4)
        torch.cuda.empty_cache()


def get_args():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument("--do_train", action='store_true', help="Whether to run training")
    parser.add_argument("--do_val", action='store_true', help="Whether to run validation")
    parser.add_argument("--do_predict", action='store_true', help="Whether to run prediction")

    parser.add_argument("--bert_model", default="bert-base-cased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-cased, bert-large-cased.")
    parser.add_argument("--config_path", default=None, type=str, help="Bert config file path.")
    parser.add_argument("--ckpts_dir",
                        required=True,
                        type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--output_dir",
                        required=True,
                        type=str,
                        help="The output directory where the model predictions and loss curves.")
    parser.add_argument("--log_file",
                        default="run.log",
                        type=str,
                        help="The output directory where the log will be written.")
    parser.add_argument("--model_recover_path",
                        default=None,
                        type=str,
                        help="The file of fine-tuned pretraining model.")
    parser.add_argument("--detectron_dir",
                        required=True,
                        type=str,
                        help="The folder holding pretrained detectron weights and bias, "
                             "see https://github.com/LuoweiZhou/VLP#-misc and "
                             "https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/detectron_weights.tar.gz"
                        )

    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=64,
                        type=int,
                        help="Batch size. This is the actual batch size during training."
                             "Model parameters are updated every (train_batch_size * grad_accumulation_steps) batches")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--label_smoothing", default=0, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay",
                        default=0.01,
                        type=float,
                        help="The weight decay rate for Adam.")
    parser.add_argument("--finetune_decay",
                        action='store_true',
                        help="Weight decay to the original weights.")
    parser.add_argument("--num_train_epochs",
                        default=30,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=8,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--amp', action='store_true', help="Whether to use amp")
    parser.add_argument('--from_scratch', action='store_true',
                        help="Initialize parameters with random values (i.e., training from scratch).")
    parser.add_argument('--new_segment_ids', action='store_true',
                        help="Use new segment ids for bi-uni-directional LM.")
    parser.add_argument('--tokenized_input', action='store_true',
                        help="Whether the input is tokenized.")
    parser.add_argument('--len_vis_input', type=int, default=100,
                        help="The length of visual token input")
    parser.add_argument('--max_len_b', type=int, default=109,
                        help="Truncate_config: maximum length of segment B.")
    parser.add_argument('--max_len_a', type=int, default=400,
                        help="Truncate_config: maximum length of segment A.")
    parser.add_argument('--max_len_img_cxt', type=int, default=200,
                        help="maximum length of segment image context.")
    parser.add_argument('--trunc_seg', default='b',
                        help="Truncate_config: first truncate segment A/B (option: a, b).")
    parser.add_argument('--always_truncate_tail', action='store_true',
                        help="Truncate_config: Whether we should always truncate tail.")
    parser.add_argument("--mask_prob", default=0.15, type=float,
                        help="Number of prediction is sometimes less than max_pred when sequence is short.")
    parser.add_argument('--max_pred', type=int, default=3,
                        help="Max tokens of prediction.")
    parser.add_argument("--num_workers", default=4, type=int,
                        help="Number of workers for the data loader.")
    parser.add_argument('--max_position_embeddings', type=int, default=None,
                        help="max position embeddings")

    # webqa dataset
    parser.add_argument('--txt_dataset_json_path', type=str, required=True, help='Path to WebQA_train_val.json')
    parser.add_argument('--img_dataset_json_path', type=str, required=True, help='Path to WebQA_train_val.json')
    parser.add_argument('--feature_folder', type=str, required=True, help="Path to RCNN x101fpn features")
    parser.add_argument('--image_id_map_path', type=str, required=True, help="Path to image_id_map_0328.pkl")

    parser.add_argument('--use_num_samples', type=int, default=-1, help="how many samples should be loaded into memory")
    parser.add_argument('--answer_provided_by', type=str, default="img|txt")
    parser.add_argument('--use_x_distractors', action='store_true')
    parser.add_argument('--task_to_learn', type=str, default="filter|qa")

    parser.add_argument('--txt_filter_max_choices', type=int, default=16)
    parser.add_argument('--img_filter_max_choices', type=int, default=16)
    parser.add_argument('--filter_infr_log', type=str, default="filter_infr_log.txt")
    parser.add_argument("--recover_ori_ckpt", action='store_true',
                        help="Whether to load original VLP checkpoint")
    parser.add_argument("--recover_step", type=int, default=None)

    parser.add_argument('--no_img_meta', action='store_true')
    parser.add_argument('--no_img_content', action='store_true')
    parser.add_argument('--no_txt_fact', action='store_true')
    parser.add_argument('--filter_infr_th', type=str,
                        default="0.05|0.1|0.15|0.2|0.25|0.3|0.35|0.4|0.45|0.5|0.55|0.6|0.65|0.7|0.75|0.8|0.85|0.9|0.95")

    parser.add_argument("--output_suffix", default="", type=str)

    # Others for VLP
    parser.add_argument('--save_loss_curve', action='store_true')
    parser.add_argument('--split', type=str, default=['train', 'val', 'test'])
    parser.add_argument('--Qcate', type=str, default=['all'])
    parser.add_argument('--sche_mode', default='warmup_linear', type=str,
                        help="warmup_linear | warmup_constant | warmup_cosine")
    parser.add_argument('--drop_prob', default=0.1, type=float)
    parser.add_argument('--use_num_imgs', default=-1, type=int)
    parser.add_argument('--vis_mask_prob', default=0, type=float)
    parser.add_argument('--max_drop_worst_ratio', default=0, type=float)
    parser.add_argument('--drop_after', default=6, type=int)

    parser.add_argument('--relax_projection',
                        action='store_true',
                        help="Use different projection layers for tasks.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
