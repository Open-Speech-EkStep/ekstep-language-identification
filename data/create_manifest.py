import os
from pathlib import Path

train_file_name = "train_data.csv"
valid_file_name = "valid_data.csv"

if os.path.isfile(train_file_name) or os.path.isfile(valid_file_name):
    try:
        os.remove(train_file_name)
    except:
        pass
    try:
        os.remove(valid_file_name)
    except:
        pass


def create_manifest(path, label, ext, mode):
    audio_path = list(Path(path).glob("**/*." + ext))
    if mode.lower() == "train":
        file_name = train_file_name
    else:
        file_name = "valid_data.csv"
    file = open(file_name, "a+", encoding="utf-8")
    for path in audio_path:
        print(str(str(path) + "," + str(label)), file=file)


if __name__ == "__main__":
    #For train
    create_manifest(path="path_to_train_data_class_0", label=0, ext="wav", mode="train")
    create_manifest(path="path_to_train_data_class_1", label=1, ext="wav", mode="train")

    #For Valid
    create_manifest(path="path_to_valid_data_class_0", label=0, ext="wav", mode="valid")
    create_manifest(path="path_to_valid_data_class_1", label=1, ext="wav", mode="valid")

