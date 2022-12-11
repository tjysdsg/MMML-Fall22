import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    args = parser.parse_args()

    with open(args.input, encoding='utf-8') as f:
        data = json.load(f)

    res = [e for e in sorted(data, key=lambda item: item['bart_score'])]

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)


if __name__ == '__main__':
    main()
