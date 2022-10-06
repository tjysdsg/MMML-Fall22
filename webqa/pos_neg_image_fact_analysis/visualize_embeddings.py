import os
import numpy as np
import json
from argparse import ArgumentParser
from sklearn.decomposition import PCA
from webqa.rcnn_feats import RcnnFeatureLoader
from matplotlib import pyplot as plt
import seaborn as sns

sns.set(rc={'figure.figsize': (16, 10)})
sns.set_style("white")


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/ocean/projects/cis210027p/jtang1/webqa',
                        help='Path to the folder containing RCNN feature files')
    parser.add_argument('--imgid-map', type=str,
                        default='/ocean/projects/cis210027p/jtang1/webqa/image_id_map_0328.pkl',
                        help='Path to image id map file')
    parser.add_argument('--data-json', type=str,
                        default='/ocean/projects/cis210027p/jtang1/webqa/WebQA_train_val.json',
                        help='Path to the data json file')
    parser.add_argument('-N', type=int, default=500, help='Number of questions whose images are used')
    parser.add_argument('--cache-dir', type=str, default='tmp', help='Cache dir to store previously loaded embeddings')
    return parser.parse_args()


class EmbeddingClassifierDataset:
    def __init__(self, imgid_map: str, data_dir: str, cache_dir: str):
        self.feat_loader = RcnnFeatureLoader(imgid_map, data_dir)
        self.cache_dir = cache_dir

        self.cached_ids = set()
        for file in os.listdir(cache_dir):
            # if os.path.isfile(file) and file.find('.npy'):
            image_id = int(file.split('.npy')[0])
            self.cached_ids.add(image_id)

    def load(self, image_id: int, split: str) -> np.ndarray:
        """
        Cached loading
        """
        cache_file = os.path.join(self.cache_dir, f'{image_id}.npy')
        if image_id in self.cached_ids:
            return np.load(cache_file)

        embedding, _, _ = self.feat_loader.load_img_feats(image_id, split)
        embedding = embedding.numpy().ravel()

        np.save(cache_file, embedding)
        return embedding


def main():
    args = get_args()
    os.makedirs(args.cache_dir, exist_ok=True)

    embedding_dataset = EmbeddingClassifierDataset(args.imgid_map, args.data_dir, args.cache_dir)

    with open(args.data_json) as f:
        data: dict = json.load(f)

    all_image_ids = set()  # skip duplicates
    all_embeds = []
    labels = []
    topics = []
    question_types = []
    for qid, d in list(data.items())[:args.N]:
        pos = d['img_posFacts']
        neg = d['img_negFacts']
        topic = d['topic']
        qtype = d['Qcate']

        for fact in pos:
            image_id = int(fact['image_id'])
            if image_id in all_image_ids:
                continue

            try:
                embedding = embedding_dataset.load(image_id, d['split'])
            except FileNotFoundError as e:
                # print(f"Skipping error: {e}")
                continue

            all_embeds.append(embedding)
            labels.append(1)
            topics.append(topic)
            question_types.append(qtype)

        for fact in neg:
            image_id = int(fact['image_id'])
            if image_id in all_image_ids:
                continue

            try:
                embedding = embedding_dataset.load(image_id, d['split'])
            except FileNotFoundError as e:
                # print(f"Skipping error: {e}")
                continue

            all_embeds.append(embedding)
            labels.append(0)
            topics.append(topic)
            question_types.append(qtype)

    print("Running PCA")
    pca = PCA(n_components=2)
    xy = pca.fit_transform(all_embeds)

    print("Plotting")

    sns.scatterplot(x=xy[:, 0], y=xy[:, 1], hue=labels)
    plt.title("PCA of Image Embeddings vs. Positive/Negative")
    plt.savefig('embed_vs_label.jpg')
    plt.close('all')

    sns.scatterplot(x=xy[:, 0], y=xy[:, 1], hue=question_types)
    plt.title("PCA of Image Embeddings vs. Question Types")
    plt.savefig('embed_vs_question_type.jpg')
    plt.close('all')

    # FIXME: too many categories
    sns.scatterplot(x=xy[:, 0], y=xy[:, 1], hue=topics)
    plt.title("PCA of Image Embeddings vs. Question Topics")
    plt.savefig('embed_vs_topic.jpg')
    plt.close('all')


if __name__ == '__main__':
    main()
