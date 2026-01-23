import logging
import re
from pathlib import Path

# jieba静音
import jieba

jieba.setLogLevel(logging.CRITICAL)

# 加载 fast_langdetect 和 LangSplitter
import fast_langdetect
from split_lang import LangSplitter

# 预配置大模型权重路径
try:
    fast_langdetect.infer._default_detector = fast_langdetect.infer.LangDetector(fast_langdetect.infer.LangDetectConfig(
        cache_dir=Path(__file__).parent.parent.parent / "pretrained_models" / "fast_langdetect"))
except:
    pass


class LangSegmenter():
    # 基础映射表
    DEFAULT_LANG_MAP = {
        "zh": "zh",
        "yue": "zh",
        "zh-cn": "zh",
        "zh-tw": "zh",
        "ko": "ko",
        "ja": "ja",
        "en": "en",
    }

    @staticmethod
    def _has_kana(text):
        """判断是否有日文假名"""
        return bool(re.search(r'[\u3040-\u30ff]', text))

    @staticmethod
    def _has_hangul(text):
        """判断是否有韩文谚文"""
        return bool(re.search(r'[\uac00-\ud7af\u1100-\u11ff]', text))

    @staticmethod
    def getTexts(text, default_lang=""):
        if default_lang:
            return [{"lang": default_lang, "text": text}]

        if not text:
            return []

        # 1. 调用原始探测并按语言切分
        lang_splitter = LangSplitter(lang_map=LangSegmenter.DEFAULT_LANG_MAP)
        lang_splitter.merge_across_digit = False  # 数字不作为独立段
        substr = lang_splitter.split_by_lang(text=text)

        # 2. 语种修正：应用“一票否决权”
        raw_segments = []
        for item in substr:
            lang = item.lang
            content = item.text

            # --- 中日纠错 ---
            if lang == "ja":
                if not LangSegmenter._has_kana(content):
                    lang = "zh"  # 只有汉字没假名，判定为中文误报

            # --- 中韩纠错 ---
            if lang == "ko":
                if not LangSegmenter._has_hangul(content):
                    lang = "zh"  # 只有汉字没谚文，判定为中文误报

            # --- 后期清理 ---
            # 如果是繁体标记(x)或未知，一律归入 zh
            if lang not in ["zh", "en"]:
                lang = "zh"

            raw_segments.append({'lang': lang, 'text': content})

        # 3. 核心步骤：重新聚合合并 (Lang Merger)
        # 将修正后语种相同的相邻段落合并成一段
        final_list = []
        for seg in raw_segments:
            if final_list and final_list[-1]['lang'] == seg['lang']:
                # 语种相同，追加文本，不切分
                final_list[-1]['text'] += seg['text']
            else:
                # 语种不同，创建新段
                final_list.append(seg)

        return final_list


if __name__ == "__main__":
    # 测试纠错与合并
    # 这里的“测试”二字常被误判为日语，看它能否粘回前半句
    text = "这是一个中文测试，Hello, 这是一个 mixed text。"
    print(LangSegmenter.getTexts(text))