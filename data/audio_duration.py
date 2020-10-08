import subprocess
from pathlib import Path

from joblib import Parallel, delayed
from tqdm import tqdm


def req_dur(file):
    command = f"soxi -D {file}"
    time = subprocess.check_output(command, shell=True)
    time = time.decode("utf-8").split('\n')[0]
    return time


def calculate_duration(target_folder, num_workers=-1):
    total_time = []
    audio_paths = list(Path(target_folder).glob("**/*.wav"))
    print("Total_files: ", len(audio_paths))
    total_time = Parallel(n_jobs=num_workers)(delayed(req_dur)(file) for file in tqdm(audio_paths))
    total_time = [float(i) for i in total_time if i != ""]
    print(sum(total_time) / 3600, " hrs")


if __name__ == "__main__":
    calculate_duration(target_folder="/home/jupyter/", num_workers=-1)
