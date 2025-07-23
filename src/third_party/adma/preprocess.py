import os.path
import subprocess

def extract_ssl_features(folder):
    output_folder = os.path.join(folder, "hubert_feature")
    os.makedirs(output_folder, exist_ok=True)
    p = subprocess.Popen(f"python src/third_party/adma/extract_ssl_features.py \
                                    --audio-root-dir {folder} \
                                    --output-dir     {output_folder} \
                                    --file-extension .wav \
                                    --layer-index -1 \
                                    --num-workers 16 \
                                    --log-every 500", shell=True)
    outcode = p.wait()
    if (outcode):
        print("提取ssl异常")
