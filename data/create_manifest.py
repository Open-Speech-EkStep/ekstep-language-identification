import os
from pathlib import Path

if os.path.isfile("data.csv"):
    os.remove("data.csv")


def create_manifest(path, label, ext):
    audio_path = list(Path(path).glob("**/*." + ext))
    file = open("data.csv", "a+")
    for _ in audio_path:
        print(str(str(_) + "," + str(label)), file=file)


if __name__ == "__main__":
    create_manifest("/home/jupyter/lid_data/hindi", 0, "wav")
    create_manifest("/home/jupyter/lid_data/english", 1, "wav")
