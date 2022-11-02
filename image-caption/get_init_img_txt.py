import json
import jsonlines

img_ids = []

with open('../text-retriever/raw_data/WebQA_data_first_release/WebQA_test.json') as f:
    dataset = json.load(f)
    for qa_data in dataset.values():
        for img_data in qa_data['img_Facts']:
            img_ids.append(img_data['image_id'])

with open('../text-retriever/raw_data/WebQA_data_first_release/WebQA_train_val.json') as f:
    dataset = json.load(f)
    for qa_data in dataset.values():
        for img_data in qa_data['img_negFacts']:
            img_ids.append(img_data['image_id'])
        for img_data in qa_data['img_posFacts']:
            img_ids.append(img_data['image_id'])
# delete the repeated ones
img_ids = list(set(img_ids))

img_txt = []
for img_id in img_ids:
    img_txt_dict = {}
    img_txt_dict['img'] = {'img_id': img_id, 'img_file_name': str(img_id) + '.jpg'}
    img_txt_dict['txt'] = {'vit-gpt2': ''}
    img_txt.append(img_txt_dict)

img_txt_len = len(img_txt)
with jsonlines.open('./webqa_img_txt.jsonl', 'w') as f:
    f.write_all(img_txt)

#for i in range(4):
#    img_txt_part = img_txt[img_txt_len//4 * i : img_txt_len//4 * (i+1)]

#    with jsonlines.open('./webqa_img_txt_generation_part{}.jsonl'.format(i), 'w') as f:
#        f.write_all(img_txt_part)