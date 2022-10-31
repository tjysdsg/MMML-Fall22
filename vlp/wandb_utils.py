import os
import time
from typing import List, Dict


def wandb_log_loss_scores(step: int, prefix: str, loss: float, score_per_threshold: Dict[float, List[float]]):
    """
    :param step: Global step
    :param prefix: {prefix__loss, {prefix}_f1
    :param loss: Loss
    :param score_per_threshold: precision, recall, F1 per threshold
    """
    import wandb
    for th, (pr, re, f1) in score_per_threshold.items():
        wandb.log({f'{prefix} loss': loss, 'step': step, 'threshold': th})
        wandb.log({f'{prefix} precision': pr, 'step': step, 'threshold': th})
        wandb.log({f'{prefix} recall': re, 'step': step, 'threshold': th})
        wandb.log({f'{prefix} F1': f1, 'step': step, 'threshold': th})


def init_wandb(output_dir: str):
    import wandb
    # need to change to your own API when using
    os.environ['EXP_NUM'] = 'WebQA'
    os.environ['WANDB_NAME'] = time.strftime(
        '%Y-%m-%d %H:%M:%S',
        time.localtime(int(round(time.time() * 1000)) / 1000)
    )
    os.environ['WANDB_API_KEY'] = 'b6bb57b85f5b5386441e06a96b564c28e96d0733'
    os.environ['WANDB_DIR'] = output_dir
    wandb.init(project="webqa-baseline")
