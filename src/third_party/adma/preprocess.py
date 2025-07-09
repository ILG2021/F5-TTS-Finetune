import os.path
import subprocess

import click


@click.command
@click.option("--folder")
def extract_ssl_features(folder):
    output_folder = os.path.join(folder, "hubert_feature")
    os.makedirs(output_folder, exist_ok=True)
    p = subprocess.Popen(f"python src/f5_tts/scripts/extract_ssl_features.py \
                                    --audio-root-dir {folder} \
                                    --output-dir     {output_folder} \
                                    --file-extension .wav \
                                    --layer-index -1 \
                                    --num-workers 16 \
                                    --log-every 500", shell=True)
    outcode = p.wait()
    if (outcode):
        print("提取ssl异常")


if __name__ == '__main__':
    extract_ssl_features()
