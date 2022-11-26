import torch
from PIL import Image
from torchvision import transforms
from transformers.models.ofa import OFATokenizer, OFAModel
from transformers.models.ofa.generate import sequence_generator

ckpt_dir = '../OFA-tiny'
path_to_image = '../2022081110143559355.png'

mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
resolution = 256
patch_resize_transform = transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
        transforms.ToTensor(), 
        transforms.Normalize(mean=mean, std=std)
    ])


tokenizer = OFATokenizer.from_pretrained(ckpt_dir)

txt = " what does the image describe?"
inputs = tokenizer([txt], return_tensors="pt").input_ids
img = Image.open(path_to_image)
patch_img = patch_resize_transform(img).unsqueeze(0)


# using the generator of fairseq version
model = OFAModel.from_pretrained(ckpt_dir, use_cache=True)
generator = sequence_generator.SequenceGenerator(
                    tokenizer=tokenizer,
                    beam_size=5,
                    max_len_b=16, 
                    min_len=0,
                    no_repeat_ngram_size=3,
                )
data = {}
data["net_input"] = {"input_ids": inputs, 'patch_images': patch_img, 'patch_masks':torch.tensor([True])}
gen_output = generator.generate([model], data)
gen = [gen_output[i][0]["tokens"] for i in range(len(gen_output))]

# using the generator of huggingface version
model = OFAModel.from_pretrained(ckpt_dir, use_cache=False)
gen = model.generate(inputs, patch_images=patch_img, num_beams=5, no_repeat_ngram_size=3) 

print(tokenizer.batch_decode(gen, skip_special_tokens=True))