import os
import sys
import json
import argparse
from tqdm import tqdm

from third_party.text.gptsovits_text_front import convert_char_to_pinyin_sovits_f5

# 确保能找到 GPT-SoVITS 和 f5_sovits_adapter
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def process_metadata(input_file, output_file, force=False):
    """
    ID|Text -> ID|Text|JSON_Pinyin_List
    :param force: 如果为 True，则忽略断点续传，强制重写输出文件
    """
    if not os.path.exists(input_file):
        print(f"Error: 找不到输入文件 {input_file}")
        return

    processed_ids = set()

    # 1. 检查已处理的 ID (非强制模式下才执行)
    if not force and os.path.exists(output_file):
        print(f"检测到输出文件 {output_file}，正在同步进度...")
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if parts:
                    processed_ids.add(parts[0])
        print(f"已跳过 {len(processed_ids)} 条已处理数据。")
    elif force:
        print("已启用强制模式 (--force)，将重新处理所有数据并覆盖输出文件。")

    # 2. 读取输入内容
    remaining_lines = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split("|")
            item_id = parts[0]
            # 如果是 force 模式，processed_ids 为空，这里会包含所有行
            if item_id not in processed_ids:
                remaining_lines.append(line)

    if not remaining_lines:
        print("所有数据已完成或无需处理。")
        return

    print(f"开始处理 {len(remaining_lines)} 条待办数据...")

    # 3. 写入文件 (Force模式用 'w' 覆盖，普通模式用 'a' 追加)
    # 注意：'w' 模式会清空原文件内容，请确保 remaining_lines 已经读入内存
    open_mode = "w" if force else "a"

    with open(output_file, open_mode, encoding="utf-8") as f_out:
        for line in tqdm(remaining_lines, desc="Processing"):
            parts = line.split("|")
            if len(parts) < 2:
                continue

            item_id = parts[0]
            text = parts[1]

            try:
                # 调用转换函数
                if len(parts) == 3: # 已经转换过的
                    f_out.write(f"{line}\n")
                    f_out.flush()
                else:
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
    # 添加 --force 参数
    parser.add_argument("--force", action="store_true", help="强制重新转换所有数据 (会覆盖已存在的输出文件)")

    args = parser.parse_args()

    input_path = os.path.abspath(args.input)

    # 自动构造输出路径: xxx.csv -> xxx_with_pinyin.csv
    file_name, file_ext = os.path.splitext(input_path)
    output_path = f"{file_name}_with_pinyin{file_ext}"

    # 传递 force 参数
    process_metadata(input_path, output_path, force=args.force)