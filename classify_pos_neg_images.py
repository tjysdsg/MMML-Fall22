import os
import numpy as np
import json
from argparse import ArgumentParser
from sklearn.pipeline import make_pipeline
from rcnn_feats import RcnnFeatureLoader
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score


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
    parser.add_argument('-N', type=int, default=1000, help='Number of questions whose images are used')
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

    # all_image_ids = set()  # check if there is duplicates
    all_embeds = []
    labels = []
    n_positive = 0
    for qid, d in list(data.items())[:args.N]:
        pos = d['img_posFacts']
        neg = d['img_negFacts']

        for fact in pos:
            try:
                image_id = int(fact['image_id'])
                # assert image_id not in all_image_ids
                embedding = embedding_dataset.load(image_id, d['split'])
            except FileNotFoundError as e:
                # print(f"Skipping error: {e}")
                continue

            all_embeds.append(embedding)
            labels.append(1)
            n_positive += 1

        for fact in neg[:2]:  # balance positive and negative
            try:
                image_id = int(fact['image_id'])
                # assert image_id not in all_image_ids
                embedding = embedding_dataset.load(image_id, d['split'])
            except FileNotFoundError as e:
                # print(f"Skipping error: {e}")
                continue

            all_embeds.append(embedding)
            labels.append(0)

    print(f'Positive samples: {n_positive}/{len(labels)}[{n_positive / len(labels):.2f}]')

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(all_embeds, labels, test_size=0.3)

    clf = make_pipeline(StandardScaler(), SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3, n_jobs=-1))
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(f'AUC: {roc_auc_score(y_test, clf.decision_function(X_test))}')
    print(classification_report(y_test, y_pred))

    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=2)
    # xy = pca.fit_transform(all_embeds)

    # from matplotlib import pyplot as plt
    # import seaborn as sns
    # sns.scatterplot(x=xy[:, 0], y=xy[:, 1], hue=labels)
    # plt.savefig('scatter.jpg')


if __name__ == '__main__':
    main()
