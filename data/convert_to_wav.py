import os
from pathlib import Path

from joblib import Parallel, delayed
from tqdm import tqdm


def wav_convert(new_path, file_path):
    new_file_path = os.path.join(new_path, file_path.parent.stem, file_path.stem + ".wav")
    try:
        os.makedirs(os.path.join(new_path, file_path.parent.stem))
    except:
        pass
    cmd = "ffmpeg -i " + str(file_path) + " -ar 16000 -ac 1 -bits_per_raw_sample 16 -vn " + str(new_file_path)
    os.system(cmd)


def convert(target_path, new_path, is_clean, ext, num_workers=1):
    if is_clean:
        audio_path = list(Path(target_path).glob("**/clean/*." + ext))
    else:
        audio_path = list(Path(target_path).glob("**/*." + ext))
    Parallel(n_jobs=num_workers)(delayed(wav_convert)(new_path, file_path) for file_path in tqdm(audio_path))


if __name__ == "__main__":
    convert(target_path="/home/jupyter/", new_path="/home/juyter/clean/", is_clean=True, ext="wav", num_workers=-1)
