import os
import torch
import jsonlines
import argparse
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from PIL import Image
from tqdm import tqdm


def model_inference(image_paths, gen_kwargs):
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")
        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

def get_img_txt_dataset(args):
    img_txt_dataset = []
    with jsonlines.open(args.input_file_name, 'r') as input_f:
        for obj in input_f:
            img_txt_dataset.append(obj)
    return img_txt_dataset

def get_img_file_name_index_dict(args, img_txt_dataset):
    img_file_name_index_dict = {} 
    for index, img_txt in enumerate(img_txt_dataset):
        img_file_name_index_dict[img_txt['img']['img_file_name']] = index
    return img_file_name_index_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='nlpconnect/vit-gpt2-image-captioning')
    parser.add_argument('--model_type', type=str, default='vit-gpt2')
    parser.add_argument('--max_length', type=int, default=64)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--inference_batch_size', type=int, default=64)
    parser.add_argument('--image_dir', type=str, default='/projects/ogma2/users/haofeiy/utils/webqa_data/images/')
    parser.add_argument('--save_every_inference_step', type=int, default=100)
    parser.add_argument('--input_file_name', type=str, default='./webqa_img_txt_new.jsonl')
    parser.add_argument('--output_file_name', type=str, default='./webqa_img_txt_new_new.jsonl')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionEncoderDecoderModel.from_pretrained(args.model_name).to(device)
    feature_extractor = ViTFeatureExtractor.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    gen_kwargs = {"max_length": args.max_length, "num_beams": args.num_beams}

    img_txt_dataset = get_img_txt_dataset(args)
    img_file_name_index_dict = get_img_file_name_index_dict(args, img_txt_dataset)

    inference_cnt = 0
    batch_img_file_names = []
    for img_txt_pair in tqdm(img_txt_dataset):
        if img_txt_pair['txt']['vit-gpt2'] == '':
            img_file_name = img_txt_pair['img']['img_file_name']

            if os.path.exists(os.path.join(args.image_dir, img_file_name)):
                batch_img_file_names.append(img_file_name)
                if len(batch_img_file_names) == args.inference_batch_size:
                    batch_img_file_paths = [
                        os.path.join(args.image_dir, img_file_name) \
                        for img_file_name in batch_img_file_names
                    ]
                    print(len(batch_img_file_names))
                    try:
                        txts = model_inference(batch_img_file_paths, gen_kwargs)
                    except:
                        txts = ['' for _ in range(args.inference_batch_size)]
                        print('ERROR: Inference is not succesful, so I just put empty token for results.')
                    for img_file_name, txt in zip(batch_img_file_names, txts):
                        index = img_file_name_index_dict[img_file_name]
                        assert img_txt_dataset[index]['img']['img_file_name'] == img_file_name
                        assert img_txt_dataset[index]['txt'][args.model_type] == ''
                        img_txt_dataset[index]['txt'][args.model_type] = txt
                    batch_img_file_names = []

                    inference_cnt += 1
                    if inference_cnt % args.save_every_inference_step == 0:
                        with jsonlines.open(args.output_file_name, 'w') as f:
                            f.write_all(img_txt_dataset)
            else:
                print('ERROR: {} needs to exist since I need to get the generated text of this image!'.format(img_file_path))
                continue