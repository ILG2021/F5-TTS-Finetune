import os
import sys
import json
import argparse
from tqdm import tqdm
from third_party.text.gptsovits_text_front import convert_char_to_pinyin_sovits_f5

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def has_repeated_char_element(pinyin_list):
    """检测拼音列表中是否含有重复字符元素，如 '，，' '。。' 'aa' 等"""
    for item in pinyin_list:
        if len(item) > 1 and len(set(item)) == 1:
            return True
    return False


def process_metadata(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Error: 找不到输入文件 {input_file}")
        return

    # 1. 读取并处理所有行
    results = []
    skipped = 0

    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in tqdm(lines, desc="Processing"):
        parts = line.split("|")

        if len(parts) == 3:
            # source 自带拼音，直接使用
            results.append(line)

        elif len(parts) == 2:
            item_id, text = parts[0], parts[1]
            try:
                pinyin_arrays = convert_char_to_pinyin_sovits_f5([text])
                if not pinyin_arrays:
                    print(f"\n[Skip] ID: {item_id}, 转换结果为空")
                    skipped += 1
                    continue
                pinyin_list = pinyin_arrays[0]
                if has_repeated_char_element(pinyin_list):
                    print(f"\n[Skip] ID: {item_id}, 含重复字符元素: {pinyin_list}")
                    skipped += 1
                    continue
                pinyin_json = json.dumps(pinyin_list, ensure_ascii=False)
                results.append(f"{item_id}|{text}|{pinyin_json}")
            except Exception as e:
                print(f"\n[Error] ID: {item_id}, Msg: {e}")
                skipped += 1
                continue
        else:
            print(f"\n[Skip] 格式异常，跳过: {line}")
            skipped += 1

    # 2. 一次性写入
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(results) + "\n")

    print(f"\n任务完成！成功: {len(results)} 条，跳过: {skipped} 条，输出: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LJSpeech Metadata 拼音标注工具")
    parser.add_argument("input", help="输入 metadata 文件路径")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    file_name, file_ext = os.path.splitext(input_path)
    output_path = f"{file_name}_with_pinyin{file_ext}"

    process_metadata(input_path, output_path)