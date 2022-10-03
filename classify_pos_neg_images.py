import pickle
import numpy as np
import json
from argparse import ArgumentParser
from rcnn_feats import RcnnFeatureLoader


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
    return parser.parse_args()


def main():
    args = get_args()

    feat_loader = RcnnFeatureLoader(args.imgid_map, args.data_dir)

    with open(args.data_json) as f:
        data: dict = json.load(f)

    all_embeds = []
    labels = []
    for qid, d in list(data.items())[:args.N]:
        pos = d['img_posFacts']
        neg = d['img_negFacts']

        for fact in pos:
            try:
                embedding, _, _ = feat_loader.load_img_feats(int(fact['image_id']), d['split'])
            except FileNotFoundError as e:
                print(f"Skipping error: {e}")
                continue

            all_embeds.append(embedding.numpy().ravel())
            labels.append(1)

        for fact in neg:
            try:
                embedding, _, _ = feat_loader.load_img_feats(int(fact['image_id']), d['split'])
            except FileNotFoundError as e:
                print(f"Skipping error: {e}")
                continue

            all_embeds.append(embedding.numpy().ravel())
            labels.append(0)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(all_embeds, labels, test_size=0.3)

    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)
    print(classifier.score(X_test, y_test))

    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=2)
    # xy = pca.fit_transform(all_embeds)

    # from matplotlib import pyplot as plt
    # import seaborn as sns
    # sns.scatterplot(x=xy[:, 0], y=xy[:, 1], hue=labels)
    # plt.savefig('scatter.jpg')


if __name__ == '__main__':
    main()
