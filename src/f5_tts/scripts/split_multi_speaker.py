import os
import shutil
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# 尝试导入pyannote.audio（最高精度方案）
try:
    from pyannote.audio import Pipeline
    import torch

    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    print("⚠️  未安装 pyannote.audio，将使用备用方案")
    print("   推荐安装以获得最高精度：pip install pyannote.audio")

# 备用方案：speechbrain
try:
    from speechbrain.pretrained import SpeakerRecognition

    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False

# 最后备用方案：基于MFCC的方法
import librosa
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


class SpeakerDetector:
    """高精度说话人检测器"""

    def __init__(self, method='auto', hf_token=None):
        """
        初始化检测器

        参数:
            method: 'auto', 'pyannote', 'speechbrain', 'mfcc'
            hf_token: Hugging Face token (pyannote方法需要)
        """
        self.method = method
        self.hf_token = hf_token
        self.pipeline = None
        self.speaker_model = None

        if method == 'auto':
            if PYANNOTE_AVAILABLE and hf_token:
                self.method = 'pyannote'
            elif SPEECHBRAIN_AVAILABLE:
                self.method = 'speechbrain'
            else:
                self.method = 'mfcc'

        self._initialize_model()

    def _initialize_model(self):
        """初始化模型"""
        if self.method == 'pyannote' and PYANNOTE_AVAILABLE:
            try:
                print("🚀 加载 pyannote.audio 模型（最高精度）...")
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=self.hf_token
                )
                # 使用GPU加速（如果可用）
                if torch.cuda.is_available():
                    self.pipeline.to(torch.device("cuda"))
                    print("   ✓ 使用GPU加速")
                print("   ✓ 模型加载完成")
            except Exception as e:
                print(f"   ✗ 加载失败: {e}")
                print("   → 切换到备用方案")
                self.method = 'speechbrain' if SPEECHBRAIN_AVAILABLE else 'mfcc'
                self._initialize_model()

        elif self.method == 'speechbrain' and SPEECHBRAIN_AVAILABLE:
            try:
                print("🚀 加载 SpeechBrain 模型（高精度）...")
                self.speaker_model = SpeakerRecognition.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir="pretrained_models/spkrec-ecapa-voxceleb"
                )
                print("   ✓ 模型加载完成")
            except Exception as e:
                print(f"   ✗ 加载失败: {e}")
                print("   → 切换到基础方案")
                self.method = 'mfcc'

        else:
            print("🔧 使用基于MFCC的检测方法（基础方案）")

    def detect_multiple_speakers(self, audio_path, min_duration=0.5):
        """
        检测音频中是否有多个说话人

        参数:
            audio_path: 音频文件路径
            min_duration: 最小说话人持续时间（秒）

        返回:
            (is_multi_speaker, num_speakers, confidence)
        """
        if self.method == 'pyannote' and self.pipeline:
            return self._detect_pyannote(audio_path, min_duration)
        elif self.method == 'speechbrain' and self.speaker_model:
            return self._detect_speechbrain(audio_path)
        else:
            return self._detect_mfcc(audio_path)

    def _detect_pyannote(self, audio_path, min_duration):
        """使用pyannote.audio检测（最高精度）"""
        try:
            diarization = self.pipeline(audio_path, min_duration_on=min_duration)

            # 获取唯一说话人数量
            speakers = set()
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speakers.add(speaker)

            num_speakers = len(speakers)
            is_multi = num_speakers > 1
            confidence = 0.95  # pyannote精度很高

            return is_multi, num_speakers, confidence

        except Exception as e:
            print(f"   ✗ Pyannote检测失败: {e}")
            return False, 1, 0.0

    def _detect_speechbrain(self, audio_path):
        """使用SpeechBrain检测"""
        try:
            # 加载音频
            import torchaudio
            signal, fs = torchaudio.load(audio_path)

            # 将音频分成多个片段
            segment_length = int(fs * 3)  # 3秒片段
            num_segments = max(1, signal.shape[1] // segment_length)

            embeddings = []
            for i in range(num_segments):
                start = i * segment_length
                end = min(start + segment_length, signal.shape[1])
                segment = signal[:, start:end]

                if segment.shape[1] < fs:  # 至少1秒
                    continue

                # 提取说话人嵌入
                embedding = self.speaker_model.encode_batch(segment)
                embeddings.append(embedding.squeeze().cpu().numpy())

            if len(embeddings) < 2:
                return False, 1, 0.8

            embeddings = np.array(embeddings)

            # 使用DBSCAN聚类
            scaler = StandardScaler()
            embeddings_scaled = scaler.fit_transform(embeddings)

            clustering = DBSCAN(eps=0.5, min_samples=2).fit(embeddings_scaled)
            num_speakers = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)

            is_multi = num_speakers > 1
            confidence = 0.85

            return is_multi, max(1, num_speakers), confidence

        except Exception as e:
            print(f"   ✗ SpeechBrain检测失败: {e}")
            return self._detect_mfcc(audio_path)

    def _detect_mfcc(self, audio_path):
        """使用MFCC特征检测（基础方案）"""
        try:
            y, sr = librosa.load(audio_path, sr=16000)
            duration = librosa.get_duration(y=y, sr=sr)

            if duration < 1.0:
                return False, 1, 0.6

            # 提取MFCC特征
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

            # 分段分析
            n_segments = min(20, int(duration))
            segment_length = mfccs.shape[1] // n_segments

            features = []
            for i in range(n_segments):
                start = i * segment_length
                end = min(start + segment_length, mfccs.shape[1])
                if end - start > 0:
                    segment_mfcc = mfccs[:, start:end].mean(axis=1)
                    features.append(segment_mfcc)

            if len(features) < 2:
                return False, 1, 0.6

            features = np.array(features)
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            # 使用DBSCAN聚类
            clustering = DBSCAN(eps=0.8, min_samples=2).fit(features_scaled)
            labels = clustering.labels_
            num_speakers = len(set(labels)) - (1 if -1 in labels else 0)

            is_multi = num_speakers > 1
            confidence = 0.7

            return is_multi, max(1, num_speakers), confidence

        except Exception as e:
            print(f"   ✗ MFCC检测失败: {e}")
            return False, 1, 0.0


def process_audio_folder(source_folder, multi_speaker_folder="multi-speaker",
                         method='auto', hf_token=None, confidence_threshold=0.7):
    """
    处理文件夹中的所有WAV文件

    参数:
        source_folder: 源文件夹路径
        multi_speaker_folder: 多说话人文件存放的文件夹名称
        method: 检测方法 ('auto', 'pyannote', 'speechbrain', 'mfcc')
        hf_token: Hugging Face token (使用pyannote时必需)
        confidence_threshold: 置信度阈值
    """
    source_path = Path(source_folder)

    if not source_path.exists():
        print(f"❌ 文件夹不存在: {source_folder}")
        return

    # 创建多说话人文件夹
    multi_speaker_path = source_path / multi_speaker_folder
    multi_speaker_path.mkdir(exist_ok=True)

    # 获取所有WAV文件
    wav_files = list(source_path.glob("*.wav"))

    if not wav_files:
        print(f"⚠️  在 {source_folder} 中没有找到WAV文件")
        return

    print(f"\n📁 找到 {len(wav_files)} 个WAV文件")
    print(f"🔍 开始检测...\n")

    # 初始化检测器
    detector = SpeakerDetector(method=method, hf_token=hf_token)
    print(f"📊 当前使用方法: {detector.method.upper()}\n")

    multi_speaker_count = 0
    single_speaker_count = 0
    results = []

    for i, wav_file in enumerate(wav_files, 1):
        # 跳过已经在multi-speaker文件夹中的文件
        if multi_speaker_folder in str(wav_file):
            continue

        print(f"[{i}/{len(wav_files)}] 检测: {wav_file.name}")

        is_multi, num_speakers, confidence = detector.detect_multiple_speakers(str(wav_file))

        result_info = {
            'file': wav_file.name,
            'is_multi': is_multi,
            'num_speakers': num_speakers,
            'confidence': confidence
        }
        results.append(result_info)

        print(f"   说话人数: {num_speakers}, 置信度: {confidence:.2f}")

        if is_multi and confidence >= confidence_threshold:
            # 移动到multi-speaker文件夹
            dest_path = multi_speaker_path / wav_file.name
            shutil.move(str(wav_file), str(dest_path))
            print(f"   ✅ 多说话人 → 已移动到 {multi_speaker_folder}/")
            multi_speaker_count += 1
        else:
            print(f"   ℹ️  单说话人 → 保持原位置")
            single_speaker_count += 1

        print()

    print("=" * 60)
    print(f"✨ 处理完成!")
    print(f"\n📊 统计结果:")
    print(f"   - 单说话人文件: {single_speaker_count}")
    print(f"   - 多说话人文件: {multi_speaker_count}")
    print(f"   - 检测方法: {detector.method.upper()}")
    print(f"   - 多说话人文件已移动到: {multi_speaker_path}")

    # 显示详细结果
    print(f"\n📋 详细结果:")
    for result in results:
        status = "✓ 多说话人" if result['is_multi'] else "  单说话人"
        print(f"   {status} | {result['num_speakers']}人 | "
              f"置信度:{result['confidence']:.2f} | {result['file']}")

    print("=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("🎙️  高精度多说话人音频检测工具")
    print("=" * 60)

    # 选择检测方法
    print("\n可用的检测方法:")
    print("  1. auto      - 自动选择最佳方法（推荐）")
    print("  2. pyannote  - 最高精度（需要Hugging Face token）")
    print("  3. speechbrain - 高精度")
    print("  4. mfcc      - 基础方法")

    method_choice = input("\n请选择方法 (1-4，默认1): ").strip() or "1"
    method_map = {"1": "auto", "2": "pyannote", "3": "speechbrain", "4": "mfcc"}
    method = method_map.get(method_choice, "auto")

    hf_token = None
    if method in ['auto', 'pyannote']:
        print("\n💡 提示: 使用 pyannote.audio 需要 Hugging Face token")
        print("   获取方式: https://huggingface.co/settings/tokens")
        print("   然后需要接受模型许可: https://huggingface.co/pyannote/speaker-diarization-3.1")
        hf_token = input("\n请输入 Hugging Face token (可选，直接回车跳过): ").strip() or None

    folder_path = input("\n请输入WAV文件所在的文件夹路径: ").strip()

    if folder_path:
        process_audio_folder(
            folder_path,
            method=method,
            hf_token=hf_token,
            confidence_threshold=0.7
        )
    else:
        print("❌ 未输入文件夹路径")