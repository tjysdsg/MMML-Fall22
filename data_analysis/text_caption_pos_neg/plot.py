import os
import json
import h5py
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feats-dir', type=str, default='sbertFeats')
    parser.add_argument('--task', type=str, choices=['posneg','qcate'])
    parser.add_argument('--out', '-o', type=str, default="figs")
    return parser.parse_args()

def sbertLoader(feats_dir, phase, task):
    phase_h5 = h5py.File(os.path.join(feats_dir,'%s.h5'%phase),'r')
    if task == "qcate":
        all_labels = [label.decode('ASCII').split('-')[-1] for label in phase_h5.get('text_uttid')]
    if task == "posneg":
        all_labels = [label.decode('ASCII').split('-')[-2] for label in phase_h5.get('text_uttid')]
    all_embeds = np.array(phase_h5.get('text_sbert_feat'))
    return all_labels, all_embeds

if __name__ == '__main__':
    
    args = parse_args()
    task = args.task
    out_dir = os.path.join(args.out, 'pca')
    os.makedirs(out_dir, exist_ok=True)
    
    phases = ['train', 'val', 'test']
    for phase in phases:
        
        os.makedirs(os.path.join(out_dir,phase), exist_ok=True)
        phase_labels, phase_embeds = sbertLoader(args.feats_dir, phase, task)
        print("... [%s] Running PCA ..."%phase)
        pca = PCA(n_components=2)
        xy = pca.fit_transform(phase_embeds)
        
        sns.scatterplot(x=xy[:, 0], y=xy[:, 1], hue=phase_labels)
        if task == "qcate":
            plt.title("%s - PCA of Image Caption Embeddings vs. Question Category"%phase)
        elif task == "posneg":
            plt.title("%s - PCA of Image Caption Embeddings vs. Positive/Negative"%phase)
        plt.savefig(f'{out_dir}/{phase}/{phase}_{task}_pca.png', dpi=150)
        plt.close('all')