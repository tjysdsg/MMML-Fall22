import os
import torch
import jsonlines
import argparse
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from PIL import Image
from tqdm import tqdm


def predict(image_paths, gen_kwargs):
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='nlpconnect/vit-gpt2-image-captioning')
    parser.add_argument('--max_length', type=int, default=64)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--inference_batch_size', type=int, default=64)
    parser.add_argument('--image_dir', type=str, default='/projects/ogma2/users/haofeiy/utils/webqa_data/images/')
    parser.add_argument('--save_every_inference_step', type=int, default=1000)
    parser.add_argument('--input_file_name', type=str, default='./webqa_img_txt_generation.jsonl')
    parser.add_argument('--output_file_name', type=str, default='./webqa_img_txt_generation_new.jsonl')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionEncoderDecoderModel.from_pretrained(args.model_name).to(device)
    feature_extractor = ViTFeatureExtractor.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    gen_kwargs = {"max_length": args.max_length, "num_beams": args.num_beams}

    with jsonlines.open(args.input_file_name, 'r') as input_f:
        img_txt_dataset = [obj for obj in input_f]

    img_id_dict = {} 
    for line_id, img_txt in enumerate(img_txt_dataset):
        img_id_dict[img_txt['image_id']] = line_id

    batch_img_file_path = [] 
    batch_img_id = []
    inference_cnt = 0
    for img_txt_data in tqdm(img_txt_dataset):
        if img_txt_data['text_generation']['vit-gpt2'] != '':
            continue
        else:
            img_id = img_txt_data['image_id']
            img_file_name = img_id + '.jpg'
            img_file_path = os.path.join(args.image_dir, img_file_name)
            if os.path.exists(img_file_path):
                batch_img_file_path.append(img_file_path)
                batch_img_id.append(img_id)
                if len(batch_img_file_path) == args.inference_batch_size:
                    preds = predict(batch_img_file_path, gen_kwargs)
                    for img_id, pred in zip(batch_img_id, preds):
                        assert img_txt_dataset[img_id_dict[img_id]]['text_generation']['vit-gpt2'] == ''
                        assert img_txt_dataset[img_id_dict[img_id]]['image_id'] == img_id
                        img_txt_dataset[img_id_dict[img_id]]['text_generation']['vit-gpt2'] = pred
                    batch_img_file_path = []
                    batch_img_id = []

                    inference_cnt += 1
                    if inference_cnt % args.save_every_inference_step:
                        with jsonlines.open(args.output_file_name, 'w') as f:
                            f.write_all(img_txt_dataset)

            else:
                raise ValueError('{} needs to exist since I need to get the generated text of this image!'.format(img_file_path))