import os
import json
import random

# 定义问题模板
aes_questions = [
    "What is the overall aesthetic quality of this advertisement image?",
    "How visually appealing does this advertisement look?",
    "What is your evaluation of the ad’s artistic quality?",
    "How attractive is the color composition and visual balance of this ad?",
    "How professional does the visual quality of the ad appear?",
    "What is the level of creativity and artistic design in this advertisement?",
    "How would you rate the visual clarity and sharpness of this ad?",
    "What is the overall impression of this ad’s visual aesthetics?"
]

ads_questions = [
    "What is the overall advertising effectiveness of this image?",
    "How clear is the promotional purpose of this advertisement?",
    "What is your evaluation of the ad’s core message delivery?",
    "How effective are the textual and visual elements in conveying the brand?",
    "What is the clarity level of the product’s key selling point in this ad?",
    "How well does this advertisement connect with its intended audience?",
    "What is the degree of relevance between the ad’s visuals and the product/service?",
    "How would you rate the overall persuasiveness of this advertisement?"
]

def build_qa_from_json(input_dir, output_file):
    all_qas = []
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as f:
                data = json.load(f)
                
                # 生成美学问题
                aes_question = random.choice(aes_questions)
                all_qas.append({
                    "id": data["id"],
                    "image_id": data["image_id"],
                    "file_path": data["file_path"],
                    "question": aes_question,
                    "answer": data["aes_score"],
                    "question_type": "aes"
                })
                
                # 生成广告属性问题
                ads_question = random.choice(ads_questions)
                all_qas.append({
                    "id": data["id"],
                    "image_id": data["image_id"],
                    "file_path": data["file_path"],
                    "question": ads_question,
                    "answer": data["ads_score"],
                    "question_type": "ads"
                })
    
    # 保存最终结果
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_qas, f, ensure_ascii=False, indent=2)

# 使用示例
build_qa_from_json("step1", "qa_dataset.json")
