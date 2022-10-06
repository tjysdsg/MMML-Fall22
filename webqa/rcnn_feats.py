import pickle
import io
import os
import torch


class TorchCPUUnpickler(pickle.Unpickler):
    """
    Load pickled torch CUDA objects into CPU
    """

    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


class RcnnFeatureLoader:
    def __init__(self, imgid_map: str, data_dir: str):
        with open(imgid_map, "rb") as f:
            self.imgid_map = pickle.load(f)

        self.data_dir = data_dir

    def load_img_feats(self, embed_id: int, split: str) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """Load image RCNN features"""
        filename = os.path.join(self.data_dir, split, f'{self.imgid_map[embed_id]}.pkl')
        with open(filename, 'rb') as f:
            feature = TorchCPUUnpickler(f).load()

        # pred_classes, image_size, num_instances
        img = feature['fc1_features'].detach().cpu().float()
        class_feats = feature['cls_features'].detach().cpu().float()
        boxes = feature['pred_boxes'].detach().cpu()
        scores = feature['scores'].detach().cpu()
        return img, class_feats, boxes, scores
