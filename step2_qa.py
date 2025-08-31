import os
import json
import random
import cv2
from tqdm import tqdm
from collections import defaultdict
from datasets import Dataset, DatasetDict, Features, Value, Image

def create_local_dataset(train_data, output_dir):
    dataset = DatasetDict({
        'train': Dataset.from_dict(
            train_data,
            features=Features({
                'id': Value('string'),
                'problem': Value('string'),
                'solution': Value('string'),
                'image': Image(),
                'img_height': Value('int64'),
                'img_width': Value('int64'),
                'resized_height': Value('int64'),
                'resized_width': Value('int64')
            })
        )
    })
    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(output_dir)
    print(f"✅ 数据集已保存到: {output_dir}")
    return dataset

def split_by_ratio(data_list, ratio=(8,1,1), seed=2025):
    random.Random(seed).shuffle(data_list)
    n = len(data_list)
    n_train = int(n * ratio[0] / sum(ratio))
    n_val = int(n * ratio[1] / sum(ratio))
    n_test = n - n_train - n_val
    return {
        "train": data_list[:n_train],
        "val": data_list[n_train:n_train+n_val],
        "test": data_list[n_train+n_val:]
    }

if __name__ == "__main__":
    json_path = "qa_dataset.json"      # 你的问答 json
    output_base = "data/SegZero_qualityt_qa_split"
    resize_hw = 768##576
    debug = False #True
    debug_n = 100

    # 1. 加载数据并按 "question_type" & "answer" 二级分组
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 二级分组
    group2list = defaultdict(list) # (question_type, answer) -> list
    for item in data:
        qtype = item.get('question_type', 'unknown')
        answer = item.get('answer', 'unknown')  # 直接用 answer 分组
        group2list[(qtype, answer)].append(item)

    # 3. 各组内做 train/val/test
    split_data_dict = {"train": [], "val": [], "test": []}
    split_count_dict = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})

    for (qtype, answer), items in group2list.items():
        # 可选调试（只采样每类 debug_n 条）
        items_ = random.sample(items, min(debug_n, len(items))) if debug and len(items) > debug_n else items
        split_dict = split_by_ratio(items_, ratio=(8,1,1))
        for split, sublist in split_dict.items():
            for item in sublist:
                item['split'] = split
                split_data_dict[split].append(item)
                split_count_dict[(qtype, answer)][split] += 1

    # 4. 保存每个 split
    for split in ["train", "val", "test"]:
        split_items = split_data_dict[split]
        if len(split_items) == 0:
            print(f"⚠️ split={split} 没有样本，跳过保存。")
            continue

        id_list, problem_list, solution_list = [], [], []
        image_list, img_height_list, img_width_list = [], [], []
        resized_height_list, resized_width_list = [], []

        for item in tqdm(split_items, desc=f"处理{split}", total=len(split_items)):
            img_path = item.get('path') or item.get('image_path') or item.get('file_path')
            if not img_path or not os.path.isfile(img_path):
                print(f"❌ 图片不存在: {img_path}")
                continue
            image = cv2.imread(img_path)
            if image is None:
                print(f"❌ 读取失败: {img_path}")
                continue
            height, width = image.shape[:2]
            resized_image = cv2.resize(image, (resize_hw, resize_hw), interpolation=cv2.INTER_AREA)

            id_list.append(str(item.get('id', '')))
            problem_list.append(str(item.get('question', '')))
            solution_obj = {
                "answer": item.get('answer', None),
                "answer_type": item.get('question_type', None)
            }
            solution_list.append(json.dumps(solution_obj, ensure_ascii=False))
            image_list.append(resized_image)
            img_height_list.append(height)
            img_width_list.append(width)
            resized_height_list.append(resize_hw)
            resized_width_list.append(resize_hw)

        train_data = {
            'id': id_list,
            'problem': problem_list,
            'solution': solution_list,
            'image': image_list,
            'img_height': img_height_list,
            'img_width': img_width_list,
            'resized_height': resized_height_list,
            'resized_width': resized_width_list
        }

        out_dir = f"{output_base}/{split}"
        create_local_dataset(train_data=train_data, output_dir=out_dir)

    # 5. 打印每个question_type + answer等级在各split的数量
    print("\n=== 各 question_type + answer 等级的 train/val/test 数量 ===")
    print("{:<12} {:<10} {:>6} {:>6} {:>6} {:>6}".format("question_type", "answer", "train", "val", "test", "total"))
    all_groups = list(group2list.keys())
    for (qtype, answer) in all_groups:
        tr, va, te = split_count_dict[(qtype, answer)]["train"], split_count_dict[(qtype, answer)]["val"], split_count_dict[(qtype, answer)]["test"]
        print("{:<12} {:<10} {:>6} {:>6} {:>6} {:>6}".format(qtype, answer, tr, va, te, tr+va+te))
    print("=============================================================")
