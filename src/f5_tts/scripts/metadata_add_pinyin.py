import os
import sys
import json
import argparse
from tqdm import tqdm

from third_party.text.gptsovits_text_front import convert_char_to_pinyin_sovits_f5

# 确保能找到 GPT-SoVITS 和 f5_sovits_adapter
sys.path.append(os.path.dirname(os.path.abspath(__file__)))



def process_metadata(input_file, output_file):
    """
    ID|Text -> ID|Text|JSON_Pinyin_List
    """
    if not os.path.exists(input_file):
        print(f"Error: 找不到输入文件 {input_file}")
        return

    # 1. 检查已处理的 ID (断点续传)
    processed_ids = set()
    if os.path.exists(output_file):
        print(f"检测到输出文件 {output_file}，正在同步进度...")
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if parts:
                    processed_ids.add(parts[0])
        print(f"已跳过 {len(processed_ids)} 条已处理数据。")

    # 2. 读取输入内容
    with open(input_file, "r", encoding="utf-8") as f:
        remaining_lines = []
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split("|")
            item_id = parts[0]
            if item_id not in processed_ids:
                remaining_lines.append(line)

    if not remaining_lines:
        print("所有数据已完成或无需处理。")
        return

    print(f"开始处理 {len(remaining_lines)} 条待办数据...")

    # 3. 追加模式执行
    with open(output_file, "a", encoding="utf-8") as f_out:
        for line in tqdm(remaining_lines, desc="Processing"):
            parts = line.split("|")
            if len(parts) < 2:
                continue

            item_id = parts[0]
            text = parts[1]

            try:
                pinyin_arrays = convert_char_to_pinyin_sovits_f5([text])
                if pinyin_arrays:
                    pinyin_json = json.dumps(pinyin_arrays[0], ensure_ascii=False)
                    f_out.write(f"{item_id}|{text}|{pinyin_json}\n")
                    f_out.flush()
            except Exception as e:
                print(f"\n[Error] ID: {item_id}, Msg: {e}")
                continue

    print(f"\n任务完成！输出文件: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LJSpeech Metadata 拼音标注工具 (支持断点续传)")
    parser.add_argument("input", help="输入 metadata 文件路径 (例如 metadata.csv)")

    args = parser.parse_args()

    input_path = os.path.abspath(args.input)

    # 自动构造输出路径: xxx.csv -> xxx_with_pinyin.csv
    file_name, file_ext = os.path.splitext(input_path)
    output_path = f"{file_name}_with_pinyin{file_ext}"

    process_metadata(input_path, output_path)
