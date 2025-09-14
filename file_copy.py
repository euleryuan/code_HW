# build_subset.py
import json, os, shutil, random, argparse
from pathlib import Path
from collections import defaultdict

ALLOWED_QT = {"ads", "aes"}
ALLOWED_ANS = ["Poor", "Bad", "Fair", "Good", "Excellent"]

def get_key(d, *candidates, default=None):
    """Try multiple key spellings; return first existing."""
    for k in candidates:
        if k in d:
            return d[k]
    return default

def resolve_image_path(fp: str, json_path: Path) -> Path:
    """
    尽量把 file_path 解析为可读文件：
    1) 如果 fp 是绝对路径且存在 → 返回
    2) 尝试按相对 JSON 文件所在目录解析
    3) 尝试按当前工作目录解析
    """
    if not fp:
        return None
    p = Path(fp)
    if p.is_file():
        return p
    # 相对 JSON 所在目录
    jp = (json_path.parent / fp).resolve()
    if jp.is_file():
        return jp
    # 相对当前工作目录
    cp = Path.cwd() / fp
    if cp.is_file():
        return cp.resolve()
    return None

def main():
    parser = argparse.ArgumentParser(description="Build ads/aes subset and copy images by answer buckets.")
    parser.add_argument("--json", required=True, help="Path to qa_datase.json")
    parser.add_argument("--out-root", default="./dataset", help="Output dataset root folder")
    parser.add_argument("--per-class", type=int, default=2000, help="Max samples per (question_type, answer)")
    parser.add_argument("--out-json", default="qa_dataset_subset.json", help="Output JSON file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    random.seed(args.seed)

    json_path = Path(args.json).expanduser().resolve()
    if not json_path.is_file():
        raise FileNotFoundError(f"JSON not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON should be a list of QA items.")

    # 1) 过滤出 ads/aes + 有效答案 + 可读图片路径
    buckets = defaultdict(list)  # (qt, ans) -> list of (item, src_path)
    missing_img, skipped_qt, skipped_ans = 0, 0, 0

    for item in data:
        qt = str(get_key(item, "question_type", "quetion_type", default="")).strip().lower()
        if qt not in ALLOWED_QT:
            skipped_qt += 1
            continue

        ans = get_key(item, "answer", default=None)
        if ans not in ALLOWED_ANS:
            skipped_ans += 1
            continue

        fp = get_key(item, "file_path", default=None)
        src = resolve_image_path(fp, json_path)
        if src is None:
            missing_img += 1
            continue

        buckets[(qt, ans)].append((item, src))

    # 2) 采样并复制
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    output_records = []
    copy_fail = 0

    for qt in sorted(ALLOWED_QT):
        for ans in ALLOWED_ANS:
            pairs = buckets.get((qt, ans), [])
            if not pairs:
                print(f"[WARN] No items for ({qt}, {ans}).")
                continue

            # 随机打乱取上限
            random.shuffle(pairs)
            chosen = pairs[: args.per_class]

            # 目标目录：dataset/qt/ans
            dst_dir = out_root / qt / ans
            dst_dir.mkdir(parents=True, exist_ok=True)

            for item, src in chosen:
                # 构建稳定的目标文件名，避免重名覆盖：优先用 image_id 或 id
                image_id = get_key(item, "image_id", default=None)
                id_ = get_key(item, "id", default=None)
                base_name = Path(src).stem
                # 生成文件名：优先 image_id，其次 id，否则原名
                name_tag = image_id or id_ or base_name
                # 保留原扩展名
                dst_path = dst_dir / f"{name_tag}{src.suffix}"

                # 如果重名，追加短随机后缀
                cnt = 1
                while dst_path.exists():
                    dst_path = dst_dir / f"{name_tag}_{cnt}{src.suffix}"
                    cnt += 1

                try:
                    shutil.copy2(src, dst_path)
                except Exception as e:
                    copy_fail += 1
                    # 跳过复制失败的
                    continue

                # 写入输出 JSON 记录（仅保留要求的字段，并更新 file_path 为绝对路径）
                new_item = {
                    "id": get_key(item, "id", default=None),
                    "image_id": get_key(item, "image_id", default=None),
                    # 拼写容错：保留旧字段名 'quetion'
                    "quetion": get_key(item, "quetion", "question", default=None),
                    "answer": ans,
                    "question_type": qt,
                    "file_path": str(dst_path.resolve()),
                }
                output_records.append(new_item)

            print(f"[OK] ({qt}, {ans}) 选取 {len(chosen)} / 可用 {len(pairs)} → 已复制到 {dst_dir}")

    # 3) 保存新 JSON
    out_json = Path(args.out_json).resolve()
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output_records, f, ensure_ascii=False, indent=2)

    # 4) 摘要
    print("\n===== Summary =====")
    print(f"Input JSON: {json_path}")
    print(f"Output root: {out_root}")
    print(f"Output JSON: {out_json}  (records: {len(output_records)})")
    print(f"Skipped (question_type not in {ALLOWED_QT}): {skipped_qt}")
    print(f"Skipped (answer not in {ALLOWED_ANS}): {skipped_ans}")
    print(f"Missing/Unreadable images: {missing_img}")
    print(f"Copy failures: {copy_fail}")

if __name__ == "__main__":
    main()
