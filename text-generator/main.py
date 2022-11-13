import os
import torch
import argparse
import time
import csv
import shutil
import evaluate
import wandb
import json
import jsonlines
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm, trange
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import Adafactor, get_cosine_schedule_with_warmup
from transformers.optimization import AdafactorSchedule
from dataset import WebQADataset, WebQATestDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import DistributedDataParallel as DDP
from val_eval import webqa_acc_approx, webqa_fl

import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(
    level=logging.DEBUG, 
    filename='./webqa.log', 
    filemode='w', 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def set_seed(args):
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def set_wandb(args):
    if args.use_wandb:
        # need to change to your own API when using
        #os.environ['WANDB_API_KEY'] = '972035264241fb0f6cc3cab51a5d82f47ca713db'
        wandb.init(project="WebQA", name=args.timestamp, config=args, dir='../WebQA_tmp')
    return


def attach_dataloader(args, tokenizer):
    loader_dict = {}
    if args.train:
        train_dataset = WebQADataset(args, tokenizer, 'train')
        val_dataset = WebQADataset(args, tokenizer, 'val')
        train_dataloader = DataLoader(
            train_dataset,  
            batch_size=args.train_batch_size, 
            shuffle=True, 
            collate_fn=train_dataset.collate_fn
        )
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=args.val_batch_size, 
            shuffle=True, 
            collate_fn=val_dataset.collate_fn
        )
        loader_dict['train'] = train_dataloader
        loader_dict['val'] = val_dataloader

    if args.inference:
        test_dataset = WebQATestDataset(args, tokenizer)
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=args.test_batch_size, 
            shuffle=False, 
            collate_fn=test_dataset.collate_fn
        )
        loader_dict['test'] = test_dataloader
    
    return loader_dict


def attach_tokenizer(args):
    return T5Tokenizer.from_pretrained(args.model_name)


def attach_model(args):
    if args.model_type == 't5-small' or args.model_type == 't5-base':
        model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    else:
        raise ValueError('Invalid model type')
    device = torch.device('cuda')
    model.to(device)

    if args.load_from_ckpt:
        model_dict = torch.load(args.load_from_ckpt)
        model.load_state_dict(model_dict, strict=False)
    return model


def attach_optimizer(args, model):
    if args.optimizer_type == 'adamw':
        optimizer = AdamW(
            model.parameters(), 
            lr=args.learning_rate, 
            correct_bias=False
        )
    elif args.optimizer_type == 'adafactor':
        optimizer = Adafactor(
            model.parameters(), 
            lr=args.learning_rate,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
    else:
        raise ValueError('Invalid optimizer type')
    return optimizer


def attach_scheduler(args, optimizer, train_dataloader):
    train_steps_per_epoch = len(train_dataloader)
    total_training_steps = args.num_epochs * train_steps_per_epoch // args.gradient_accumulation_step
    total_warmup_steps = total_training_steps // 10
    if args.scheduler_type == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_warmup_steps,
            num_training_steps=total_training_steps,
        )
        return scheduler
    elif args.scheduler_type == 'adafactorschedule':
        #scheduler = AdafactorSchedule(optimizer)
        scheduler = None
        return scheduler
    else:
        raise ValueError('Invalid scheduler type')


def save_model(best_ckpt_name, metric, best_metric):
    eps = 1e-5
    if (metric['acc']*metric['fl']) > ((best_metric['acc']*best_metric['fl']) + eps):
        if best_ckpt_name is not None:
            os.remove(os.path.join(args.ckpt_save_dir,best_ckpt_name))
        best_ckpt_name = 'best_{}4{}_f1_{}_recall_{}_acc_{}_fl_{}_{}.ckpt'.format(
            args.model_type, 
            args.task, 
            round(metric['f1'],3), 
            round(metric['recall'],3),
            round(metric['acc'],3),
            round(metric['fl'],3),
            args.timestamp
        )
        output_model_file = os.path.join(args.ckpt_save_dir, best_ckpt_name)
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), output_model_file)
        best_metric['f1'] = metric['f1']
        best_metric['recall'] = metric['recall']
        best_metric['acc'] = metric['acc']
        best_metric['fl'] = metric['fl']
    return best_ckpt_name, best_metric


def save_final_model(best_ckpt_name):
    src_file = os.path.join(args.ckpt_save_dir, best_ckpt_name)
    tgt_file = os.path.join(args.ckpt_save_dir, 'best_{}4{}.ckpt'.format(args.model_type, args.task))
    shutil.copy(src_file, tgt_file)
    return


def validate(args, val_dataloader, model, tokenizer):
    model.eval()

    refs = []
    preds = []
    Qcates = []
    with torch.no_grad():
        val_iter = tqdm(val_dataloader, desc="Validation", disable=0)
        for batch in val_iter:
            pred_tokens = model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                decoder_start_token_id=tokenizer.pad_token_id,
                max_length=args.decoding_max_length,
                num_beams=5,
            )
            ref_tokens = batch['labels']
            ref_tokens[ref_tokens == -100] = tokenizer.pad_token_id
            pred = tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)
            ref = tokenizer.batch_decode(ref_tokens, skip_special_tokens=True)
            preds += pred
            refs += ref
            Qcates += batch['Qcates']
    assert len(preds) == len(refs) == len(Qcates)

    f1 = 0
    recall = 0

    eval_metric = {
        'color': [], 
        'shape': [], 
        'number': [], 
        'YesNo': [],
        'number': [],
        'text': [],
        'Others': [],
        'choose': [],
        'f1': [],
        'recall': [],
        'acc': [],
        'fl': 0,
    }
    eval_metric['fl'] = webqa_fl(preds, refs)['fl']
    for pred, ref, Qcate in zip(preds, refs, Qcates):
        eval_output = webqa_acc_approx(pred, ref, Qcate)['acc_approx']
        eval_metric[Qcate].append(eval_output)
        if Qcate in ['color', 'shape', 'number', 'YesNo']:
            eval_metric['f1'].append(eval_output)
        else:
            eval_metric['recall'].append(eval_output)
        eval_metric['acc'].append(eval_output)
    for key, value in eval_metric.items():
        if key == 'fl':
            continue
        if len(eval_metric[key]) == 0:
            eval_metric[key] = 0
        else:
            eval_metric[key] = sum(eval_metric[key]) / len(eval_metric[key])

    return eval_metric


def train(args, model, tokenizer):
    best_ckpt_name = None
    best_metric = {'f1': 0, 'recall': 0, 'acc': 0, 'fl': 0)}
    step = 0
    iteration = 0
    logging.info('=====begin loading dataset====')
    loaders = attach_dataloader(args, tokenizer)
    logging.info('=====end loading dataset====')
    train_dataloader = loaders['train']
    val_dataloader = loaders['val']
    
    optimizer = attach_optimizer(args, model)
    scheduler = attach_scheduler(args, optimizer, train_dataloader)
    model.train()

    step_losses = []
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_fp16)
    train_iter = trange(args.num_epochs, desc="Epoch", disable=-1)
    for epoch in train_iter:
        epoch_iter = tqdm(train_dataloader, desc="Iteration", disable=0)
        for batch in epoch_iter:
            model.train()
            with torch.cuda.amp.autocast(enabled=args.use_fp16):
                outputs = model(
                    input_ids=batch['input_ids'],
                    labels=batch['labels'],
                    decoder_attention_mask=batch['decoder_attention_mask'],
                    attention_mask=batch['attention_mask'],
                )
                loss = outputs.loss
                scaler.scale(loss).backward()
                step_losses.append(loss.item())

            iteration += 1
            if iteration % args.gradient_accumulation_step == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                #scheduler.step()
                
                step += 1
                if step % args.evaluation_steps == 0:
                    metric = validate(args, val_dataloader, model, tokenizer)
                    best_ckpt_name, best_metric = save_model(best_ckpt_name, metric, best_metric)
                    if args.use_wandb:
                        wandb.log({'eval_f1': metric['f1'], 'step': step})
                        wandb.log({'eval_recall': metric['recall'], 'step': step})
                        wandb.log({'eval_acc': metric['acc'], 'step': step})
                        wandb.log({'eval_FL': metric['fl'], 'step': step})
                        wandb.log({'eval_qa': metric['acc']*metric['fl'], 'step': step})
                    if args.use_logger:
                        logging.info('eval f1 : {}'.format(metric['f1']))
                        logging.info('eval recall : {}'.format(metric['recall']))
                        logging.info('eval acc : {}'.format(metric['acc']))
                        logging.info('eval fl : {}'.format(metric['fl']))
                        logging.info('eval qa : {}'.format(metric['acc']*metric['fl']))

                if args.use_wandb:
                    wandb.log({'train loss': sum(step_losses)/len(step_losses), 'step': step})
                    wandb.log({'learning rate': args.learning_rate, 'step': step})
                if args.use_logger:
                    logging.info('train loss : {}'.format(sum(step_losses)/len(step_losses)))
                step_losses = []
                    
    save_final_model(best_ckpt_name)
    return


def inference(args, model, tokenizer):
    loader = attach_dataloader(args, tokenizer)
    test_dataloader = loader['test']

    model.load_state_dict(
        torch.load(
            os.path.join(args.ckpt_save_dir, 'best_{}4{}.ckpt'.format(args.model_type, args.task))
    ))

    model.eval()

    preds = []
    qids = []
    with torch.no_grad():
        test_iter = tqdm(test_dataloader, desc="Testing", disable=0)
        for batch in test_iter:
            pred_tokens = model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                decoder_start_token_id=tokenizer.pad_token_id,
                max_length=args.decoding_max_length,
                num_beams=5,
            )
            qids += batch['qids']
            pred = tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)
            preds += pred
    assert len(preds) == len(qids)

    with open(args.retrieved_result_file, 'r') as f1, open(args.result_file, 'w') as f2:
        retrieved_result = json.load(f1)
        for qid, pred in zip(qids, preds):
            retrieved_result[qid]['answer'] = pred
        json.dump(retrieved_result, f2, indent=4)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--timestamp', type=str, default=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(round(time.time()*1000))/1000)))
    parser.add_argument('--model_type', type=str, default='t5-base', choices=['t5-small', 't5-base'])
    parser.add_argument('--model_name', type=str, default='t5-base')
    parser.add_argument('--have_cached_dataset', action='store_true')
    parser.add_argument('--cache_dir', type=str, default='../cache')
    parser.add_argument('--train_file', type=str, default='../data/WebQA_full_data/train.jsonl')
    parser.add_argument('--val_file', type=str, default='../data/WebQA_full_data/val.jsonl')
    parser.add_argument('--test_file', type=str, default='../data/WebQA_test_data/retrieved_test.jsonl')
    parser.add_argument('--retrieved_result_file', type=str, default='../data/WebQA_test_data/submission_test.json')
    parser.add_argument('--result_file', type=str, default='../data/WebQA_test_data/final_submission.json')
    parser.add_argument('--output_file', type=str, default='../data/submission.jsonl')
    parser.add_argument('--task', type=str, default='webqa-finetune', choices=['webqa-finetune'])
    parser.add_argument('--load_from_ckpt', type=str, default=None)
    parser.add_argument('--ckpt_save_dir', type=str, default='../checkpoints/')
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--gradient_accumulation_step', type=int, default=4)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--test_batch_size', type=int, default=8)
    parser.add_argument('--encoder_max_length', type=int, default=512)
    parser.add_argument('--decoder_max_length', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--optimizer_type', type=str, default='adafactor')
    parser.add_argument('--scheduler_type', type=str, default='adafactorschedule')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset', type=str, default='webqa')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--evaluation_steps', type=int, default=50)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--use_logger', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--use_fp16', action='store_true')
    parser.add_argument('--decoding_max_length', type=int, default=30)
    args = parser.parse_args()
    set_seed(args)
    set_wandb(args)


    tokenizer = attach_tokenizer(args)
    model = attach_model(args)

    if args.train:
        train(args, model, tokenizer)
    elif args.inference:
        inference(args, model, tokenizer)
