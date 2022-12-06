import os
import torch
import argparse
import time
import json
import shutil
import evaluate
import torch.distributed as dist
from tqdm import tqdm
from modeling_ofa import OFAModel
from torch.utils.data import DataLoader
from transformers.models.ofa import OFATokenizer
from transformers import get_cosine_schedule_with_warmup
from dataset import WebQADataset, WebQATestDataset


def load_dataset(args, tokenizer):
    '''
    loading datasets, return a dictionary of dataloaders
    '''
    loader_dict = {}

    if args.train:
        train_dataset = WebQADataset(args, tokenizer, split='train')
        dev_dataset = WebQADataset(args, tokenizer, split='val')
        if torch.cuda.device_count() > 1 and args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
            dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
            train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, sampler=train_sampler, collate_fn=train_dataset.collate_fn, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor, pin_memory=True)
            dev_dataloader = DataLoader(dev_dataset, batch_size=args.dev_batch_size, sampler=dev_sampler, collate_fn=dev_dataset.collate_fn, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor, pin_memory=True)
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=train_dataset.collate_fn, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor, pin_memory=True)
            dev_dataloader = DataLoader(dev_dataset, batch_size=args.dev_batch_size, shuffle=True, collate_fn=dev_dataset.collate_fn, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor, pin_memory=True)
            dev_dataloader = train_dataloader
        loader_dict['train'] = train_dataloader
        loader_dict['dev'] = dev_dataloader

    if args.test:
        test_dataset = WebQATestDataset(args, tokenizer)
        test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, collate_fn=test_dataset.collate_fn, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor, pin_memory=True)
        loader_dict['test'] = test_dataloader
    
    return loader_dict


def attach_optimizer(args, model):
    '''
    attach optimizer to the model
    '''
    if args.optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        return optimizer
    else:
        raise ValueError('Invalid optimizer')


def attach_scheduler(args, optimizer, total_training_steps):
    '''
    attach lr scheduler to the model
    '''
    if args.scheduler_type == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_training_steps,
        )
        return scheduler
    else:
        raise ValueError('Invalid scheduler type')


def cross_entropy_with_logits_loss(prediction, target, logit_mask):
    # prediction: batch_size x num_choices x 2
    # target: batch_size x num_choices x 2. Targets with multiple flags look like [[0,1], [1,0], [0,1], [0,1], [0,1]] (there is no need to normalize them)
    num_choices = prediction.size(1)
    batch_size = prediction.size(0)
    lp = torch.nn.functional.log_softmax(prediction, dim=-1)  # batch_size x num_choices x 2
    # num_flags = torch.sum(target, dim=-1)
    # num_flags = torch.max(num_flags, torch.ones_like(num_flags))
    # labels = target / num_flags.unsqueeze(-1).repeat(1, prediction.size(-1))
    # m = lp * labels
    # m = torch.where(torch.isnan(m), torch.zeros_like(m), m)
    # loss = torch.sum(- m, dim=-1) * num_flags
    normalizer = torch.sum(logit_mask, dim=-1)
    m = lp * target * logit_mask.view(-1, num_choices, 1).repeat(1, 1, 2)  # target.transpose --> batch_size x num_choices x 2
    loss = (-m).view(batch_size, -1).sum(dim=-1) / (normalizer + 1e-8)
    return torch.mean(loss)


def validate(args, dev_dataloader, model):
    model.eval()
    with torch.no_grad():
        eval_losses = []
        gth_labels = []
        pred_labels = []
        for idx, data in enumerate(tqdm(dev_dataloader)):
            sources = data['sources'].to(args.device)
            prev_outputs = data['prev_outputs'].to(args.device)
            decoder_attention_mask = data['decoder_attention_mask'].to(args.device)
            constraint_masks = data['constraint_masks'].to(args.device)
            allowed_words = data['allowed_words'].to(args.device)
            labels = data['labels'].to(args.device)
            logit_mask = data['logit_mask'].to(args.device)
            if not args.without_image:
                patch_images = data['patch_images'].to(args.device)
                patch_masks = data['patch_masks'].to(args.device)

            squeezed_sources = sources.view(-1, sources.size(-1))
            squeezed_prev_outputs = prev_outputs.view(-1, prev_outputs.size(-1))
            squeezed_decoder_attention_mask = decoder_attention_mask.view(-1, decoder_attention_mask.size(-1))
            squeezed_constraint_masks = constraint_masks.view(-1, constraint_masks.size(-2), constraint_masks.size(-1))
            if not args.without_image:
                squeezed_patch_masks = patch_masks.view(-1)
                squeezed_patch_images = patch_images.view(-1, patch_images.size(-3), patch_images.size(-2), patch_images.size(-1))
            else:
                squeezed_patch_masks = None
                squeezed_patch_images = None

            assert args.choice_num % args.real_batch_size == 0
            real_logits = []
            for idx in range(sources.size(0) * args.choice_num // args.real_batch_size):
                start_idx = idx * args.real_batch_size
                end_idx = (idx + 1) * args.real_batch_size
                # TODO (haofeiyu): to confirm whether the attention mask here is actually decoder_attention_mask
                outputs = model(
                    input_ids=squeezed_sources[start_idx:end_idx], 
                    decoder_input_ids=squeezed_prev_outputs[start_idx:end_idx],
                    patch_masks=squeezed_patch_masks[start_idx:end_idx] if not args.without_image else None,
                    patch_images=squeezed_patch_images[start_idx:end_idx] if not args.without_image else None,
                    attention_mask=squeezed_decoder_attention_mask[start_idx:end_idx],
                )
                logits = outputs['logits']
                logits.masked_fill_(~squeezed_constraint_masks[start_idx:end_idx], -float('inf'))
                last_token_ids = squeezed_prev_outputs[start_idx:end_idx].ne(tokenizer.pad_token_id).sum(1, keepdim=True) - 1
                last_token_ids[last_token_ids<0] = 0
                logits = logits.gather(1, last_token_ids.unsqueeze(2).expand(-1, -1, logits.size(-1))).squeeze(1)
                logits = logits.gather(1, allowed_words.unsqueeze(0).expand(logits.size(0), -1))
                real_logits.append(logits)
            logits = torch.cat(real_logits, dim=0)
            preds = logits.view(-1, args.choice_num, args.label_num)
            refs = torch.nn.functional.one_hot(labels * logit_mask)
            refs = refs.view(-1, args.choice_num, args.label_num)
            
            # need to fix the -inf problem since the -inf will not be masked by the logit_mask
            preds.masked_fill_(~logit_mask.unsqueeze(-1).expand(-1, -1, args.label_num), 0)
            eval_loss = cross_entropy_with_logits_loss(preds, refs, logit_mask)
            softmax_logits = torch.nn.functional.softmax(logits, dim=-1)[:, 1]
            # TODO (haofeiyu): during evaluation, the extra negative sampling should not be ignored
            predictions = (softmax_logits > args.classifier_threshold).float()
            pred_labels.append(predictions.tolist())
            gth_labels.append(labels.view(-1).tolist())
            eval_losses.append(eval_loss.item()) 

    metric = evaluate.load("f1")
    true_predictions = []
    true_labels = []
    for pred, label in zip(pred_labels, gth_labels):
        true_predictions += [p for p, l in zip(pred, label) if l != -100]
        true_labels += [l for p, l in zip(pred, label) if l != -100]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    f1 = results['f1']

    print(f'validation f1 : {f1}')
    eval_loss = sum(eval_losses) / len(eval_losses)
    print(f'validation loss : {eval_loss}')
    model.train()
    return f1, eval_loss


def train(args, model, tokenizer):
    best_checkpoint_name = None
    best_eval_f1 = -float('inf')
    best_eval_loss = float('inf')
    global_step = 0
    step = 0
    print('=====begin loading dataset====')
    loaders = load_dataset(args, tokenizer)
    print('=====end loading dataset====')
    train_dataloader = loaders['train']
    dev_dataloader = loaders['dev']
    model.train()
    optimizer = attach_optimizer(args, model)
    total_training_steps = len(train_dataloader) * args.num_epochs // args.gradient_accumulation_step
    scheduler = attach_scheduler(args, optimizer, total_training_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_fp16)

    train_losses = []
    for epoch in range(args.num_epochs):
        for data in tqdm(train_dataloader):
            sources = data['sources'].to(args.device)
            prev_outputs = data['prev_outputs'].to(args.device)
            decoder_attention_mask = data['decoder_attention_mask'].to(args.device)
            constraint_masks = data['constraint_masks'].to(args.device)
            allowed_words = data['allowed_words'].to(args.device)
            labels = data['labels'].to(args.device)
            logit_mask = data['logit_mask'].to(args.device)
            if not args.without_image:
                patch_masks = data['patch_masks'].to(args.device)
                patch_images = data['patch_images'].to(args.device)

            squeezed_sources = sources.view(-1, sources.size(-1))
            squeezed_prev_outputs = prev_outputs.view(-1, prev_outputs.size(-1))
            squeezed_decoder_attention_mask = decoder_attention_mask.view(-1, decoder_attention_mask.size(-1))
            squeezed_constraint_masks = constraint_masks.view(-1, constraint_masks.size(-2), constraint_masks.size(-1))
            if not args.without_image:
                squeezed_patch_masks = patch_masks.view(-1)
                squeezed_patch_images = patch_images.view(-1, patch_images.size(-3), patch_images.size(-2), patch_images.size(-1))
            else:
                squeezed_patch_masks = None
                squeezed_patch_images = None

            with torch.cuda.amp.autocast(enabled=args.use_fp16):

                assert args.choice_num % args.real_batch_size == 0
                real_logits = []
                for idx in range(sources.size(0) * args.choice_num // args.real_batch_size):
                    start_idx = idx * args.real_batch_size
                    end_idx = (idx + 1) * args.real_batch_size
                    # TODO (haofeiyu): to confirm whether the attention mask here is actually decoder_attention_mask
                    outputs = model(
                        input_ids=squeezed_sources[start_idx:end_idx], 
                        decoder_input_ids=squeezed_prev_outputs[start_idx:end_idx],
                        patch_masks=squeezed_patch_masks[start_idx:end_idx] if not args.without_image else None,
                        patch_images=squeezed_patch_images[start_idx:end_idx] if not args.without_image else None,
                        attention_mask=squeezed_decoder_attention_mask[start_idx:end_idx],
                    )
                    logits = outputs['logits']
                    logits.masked_fill_(~squeezed_constraint_masks[start_idx:end_idx], -float('inf'))
                    last_token_ids = squeezed_prev_outputs[start_idx:end_idx].ne(tokenizer.pad_token_id).sum(1, keepdim=True) - 1
                    last_token_ids[last_token_ids<0] = 0
                    logits = logits.gather(1, last_token_ids.unsqueeze(2).expand(-1, -1, logits.size(-1))).squeeze(1)
                    logits = logits.gather(1, allowed_words.unsqueeze(0).expand(logits.size(0), -1))
                    real_logits.append(logits)

                logits = torch.cat(real_logits, dim=0)
                preds = logits.view(-1, args.choice_num, args.label_num)
                refs = torch.nn.functional.one_hot(labels * logit_mask)
                refs = refs.view(-1, args.choice_num, args.label_num)
                # need to fix the -inf problem since the -inf will not be masked by the logit_mask
                preds.masked_fill_(~logit_mask.unsqueeze(-1).expand(-1, -1, args.label_num), 0)
                loss = cross_entropy_with_logits_loss(preds, refs, logit_mask)
                loss = loss / args.gradient_accumulation_step
                scaler.scale(loss).backward()

            train_losses.append(loss.item())
            step += 1

            if step % args.gradient_accumulation_step == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

                if args.use_wandb:
                    wandb.log({'train loss': loss.item() * args.gradient_accumulation_step, 'step': global_step})
                    wandb.log({'learning rate': scheduler.get_last_lr()[0], 'step': global_step})

                global_step += 1
                if global_step % args.evaluation_steps == 0:
                    eval_f1, eval_loss = validate(args, dev_dataloader, model)
                    if args.use_wandb:
                        wandb.log({'eval_f1': eval_f1, 'step': global_step})
                        wandb.log({'eval_loss': eval_loss, 'step': global_step})
                    if args.model_chosen_metric == 'f1':
                        if eval_f1 > best_eval_f1:
                            if best_checkpoint_name is not None:
                                os.remove(best_checkpoint_name)
                                best_checkpoint_name = args.checkpoint_save_dir + 'best_{}4{}_f1_{}_{}.ckpt'.format(args.model_name, args.task, round(eval_f1*100,3), args.timestamp)
                            else:
                                best_checkpoint_name = args.checkpoint_save_dir + 'best_{}4{}_f1_{}_{}.ckpt'.format(args.model_name, args.task, round(eval_f1*100,3), args.timestamp)
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_model_file = best_checkpoint_name
                            torch.save(model_to_save.state_dict(), output_model_file)
                            best_eval_f1 = eval_f1
                    elif args.model_chosen_metric == 'loss':
                        if eval_loss < best_eval_loss:
                            if best_checkpoint_name is not None:
                                os.remove(best_checkpoint_name)
                                best_checkpoint_name = args.checkpoint_save_dir + 'best_{}4{}_loss_{}_{}.ckpt'.format(args.model_name, args.task, round(eval_loss,3), args.timestamp)
                            else:
                                best_checkpoint_name = args.checkpoint_save_dir + 'best_{}4{}_loss_{}_{}.ckpt'.format(args.model_name, args.task, round(eval_loss,3), args.timestamp)
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_model_file = best_checkpoint_name
                            torch.save(model_to_save.state_dict(), output_model_file)
                            best_eval_loss = eval_loss
                    else:
                        raise NotImplementedError
        epoch_loss = sum(train_losses) / len(train_losses)
        print(f'Epoch {epoch} loss: {epoch_loss}')

    src_file = best_checkpoint_name
    tgt_file = args.checkpoint_save_dir + 'best_{}4{}.ckpt'.format(args.model_name, args.task)
    shutil.copy(src_file, tgt_file)
    return


def test(args, model, tokenizer):
    test_results = {}
    model.eval()
    with torch.no_grad():
        model.load_state_dict(torch.load(args.checkpoint_save_dir + 'best_{}4{}.ckpt'.format(args.model_name, args.task)))
        loaders = load_dataset(args, tokenizer)
        for idx, data in enumerate(tqdm(loaders['test'])):
            sources = data['sources'].to(args.device)
            prev_outputs = data['prev_outputs'].to(args.device)
            decoder_attention_mask = data['decoder_attention_mask'].to(args.device)
            constraint_masks = data['constraint_masks'].to(args.device)
            allowed_words = data['allowed_words'].to(args.device)
            if not args.without_image:
                patch_images = data['patch_images'].to(args.device)
                patch_masks = data['patch_masks'].to(args.device)
            else:
                patch_images = None
                patch_masks = None
            source_ids = data['source_ids']
            source_types = data['source_types']
            q_ids = data['q_ids']

            outputs = model(
                input_ids=sources, 
                decoder_input_ids=prev_outputs,
                patch_images=patch_images,
                patch_masks=patch_masks,
                attention_mask=decoder_attention_mask,
            ) 
            logits = outputs['logits']
            logits.masked_fill_(~constraint_masks, -float('inf'))
            last_token_ids = prev_outputs.ne(tokenizer.pad_token_id).sum(1, keepdim=True) - 1
            logits = logits.gather(1, last_token_ids.unsqueeze(2).expand(-1, -1, logits.size(-1))).squeeze(1)
            logits = logits.gather(1, allowed_words.unsqueeze(0).expand(logits.size(0), -1))

            # need to fix the -inf problem since the -inf will not be masked by the logit_mask
            softmax_logits = torch.nn.functional.softmax(logits, dim=-1)[:, 1]
            # TODO (haofeiyu): during evaluation, the extra negative sampling should not be ignored
            predictions = (softmax_logits > args.test_classifier_threshold).float()

            assert len(predictions) == len(q_ids) == len(source_ids) == len(source_types)
            for idx in range(len(predictions)):
                prediction = predictions[idx]
                qid = q_ids[idx]
                source_id = source_ids[idx]
                if qid not in test_results.keys():
                    test_results[qid] = {"sources": [], "answer": ""}
                if prediction == 1:
                    test_results[qid]["sources"].append(source_id)

    with open("./data/WebQA_test_data/submission.json", "w") as outfile:
        json.dump(test_results, outfile)

    return


def distributed_setup(args, model):
    '''
    setup distributed training
    '''
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl')
    args.device = torch.device('cuda', args.local_rank)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_batch_size', type=int, default=1)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--without_image', action='store_true')
    parser.add_argument('--cache_dir', type=str, default='./cache', help='the location of cache file')
    parser.add_argument('--have_cached_dataset', action='store_true')
    parser.add_argument('--dataset_dir', type=str, default='./data/')
    parser.add_argument('--model_name', type=str, default='ofa-large', help='model name or path')
    parser.add_argument('--model_dir', type=str, default='./ofa-large')
    parser.add_argument('--image_dir', type=str, default='../../images')
    parser.add_argument('--train_file', type=str, default='train.jsonl', help='path to train file, jsonl for scirex, conll for sciner')
    parser.add_argument('--val_file', type=str, default='val.jsonl', help='path to dev file')
    parser.add_argument('--test_file', type=str, default='test.jsonl', help='path to test file')
    parser.add_argument('--load_from_checkpoint', type=str, default=None, help='contine finetuning based on one checkpoint')
    parser.add_argument('--model_chosen_metric', type=str, default='f1', help='choose dev checkpoint based on this metric')
    parser.add_argument('--checkpoint_save_dir', type=str, default='./checkpoints/')
    parser.add_argument('--task', type=str, default='webqa-finetune')
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_step', type=int, default=1)
    parser.add_argument('--dev_batch_size', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=4)
    parser.add_argument('--question_max_length', type=int, default=100)
    parser.add_argument('--fact_max_length', type=int, default=150)
    parser.add_argument('--answer_max_length', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--optimizer_type', type=str, default='adamw')
    parser.add_argument('--scheduler_type', type=str, default='cosine')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--label_num', type=int, default=2, help='number of labels, 1 for pos, 1 for neg')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--evaluation_steps', type=int, default=50)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--classifier_threshold', type=float, default=0.3)
    parser.add_argument('--test_classifier_threshold', type=float, default=0.3)
    parser.add_argument('--choice_num', type=int, default=16)
    parser.add_argument('--use_fp16', action='store_true')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--prefetch_factor', type=int, default=8)
    args = parser.parse_args()

    args.local_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else -1
    args.timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(round(time.time()*1000))/1000))

    if args.use_wandb:
        import wandb
        # need to change to your own API when using
        os.environ['EXP_NUM'] = 'WebQA'
        os.environ['WANDB_NAME'] = time.strftime(
            '%Y-%m-%d %H:%M:%S', 
            time.localtime(int(round(time.time()*1000))/1000)
        )
        os.environ['WANDB_API_KEY'] = '972035264241fb0f6cc3cab51a5d82f47ca713db'
        os.environ['WANDB_DIR'] = './WebQA_tmp'
        wandb.init(project="WebQA")

    tokenizer = OFATokenizer.from_pretrained(args.model_dir)
    model = OFAModel.from_pretrained(args.model_dir)
    args.vocab_size = model.config.vocab_size

    device = torch.device(args.local_rank) if args.local_rank != -1 else torch.device('cuda')
    if args.load_from_checkpoint:
        model_dict = torch.load(args.load_from_checkpoint)
        filtered_model_dict = {k: v for k, v in model_dict.items() if 'classifier' not in k}
        model_dict.update(filtered_model_dict)
        model.load_state_dict(filtered_model_dict, strict=False)
    
    model.to(device)
    
    if torch.cuda.device_count() > 1 and args.distributed:
        distributed_setup(args, model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    if args.train:
        train(args, model, tokenizer)
    elif args.test:
        test(args, model, tokenizer)
