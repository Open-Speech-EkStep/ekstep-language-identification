import os
from pathlib import Path
import sys
import torch
import torch.nn as nn
import yaml
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from sklearn.metrics import auc as au
from loaders.data_loader import SpeechDataGenerator
import numpy as np
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from utils.training import *

# Variables
tracking_uri = sys.argv[1]
valid_hindi = sys.argv[2]
# valid_english = sys.argv[2]
# valid_tamil = sys.argv[1]
# checkpoint_path = sys.argv[3]
valid_parameters = load_yaml_file("train_config.yml")["train_parameters"]
model_name = valid_parameters["model_name"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()
mlflow.set_tracking_uri(tracking_uri)
# Hyperparameters
batch_size = 128
if os.path.isfile("validation_data128.csv"):
    os.remove("validation_data128.csv")

if os.path.isfile("prediction_log128.csv"):
    os.remove("prediction_log128.csv")


def load_model():
    client = MlflowClient()
    for mv in client.search_model_versions(f"name='{model_name}'"):
        model_version = int(mv.version)

    model = mlflow.pytorch.load_model(
        model_uri=f"models:/{model_name}/{model_version}"
    )
    print(f"Model fetched with name : {model_name} and version {model_version}")
    print(model)
    return model


def create_manifest(path, label, ext):
    audio_path = list(Path(path).glob("**/*." + ext))
    file = open("validation_data128.csv", "a+")
    for _ in audio_path:
        print(str(str(_) + "," + str(label)), file=file)


# create_manifest(valid_english, int(sys.argv[5]), "wav")
create_manifest(valid_hindi, 1, "wav")
# create_manifest(valid_tamil, int(sys.argv[6]), "wav")

dataset = SpeechDataGenerator(manifest="validation_data128.csv", mode='test')

validation_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=24)

loaders = {
    'validation': validation_loader,
}

##MODEL 1

# model = resnet18(pretrained=True)
# model.fc = nn.Linear(512, 3)
# model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# model = torch.load(checkpoint_path)
model = load_model()
model.to(device)
run_id = model.metadata.run_id
print(f'The run_id is {run_id}')


def create_output(confidence_scores):
    output_dictionary = {'confidence_score': {}}
    language_map = load_yaml_file('language_map.yml')['languages']
    for key in language_map:
        output_dictionary['confidence_score'][language_map[key]] = confidence_scores[key]
    return output_dictionary


def load_yaml_file(path):
    read_dict = {}
    with open(path, 'r') as file:
        read_dict = yaml.safe_load(file)
    return read_dict


def validation(loaders, model, use_cuda):
    # if os.path.isfile(checkpoint_path):
    # print("loaded model from ",checkpoint_path)
    # model = torch.load(checkpoint_path)
    # model.load_state_dict(checkpoint['state_dict'])

    criterion = nn.CrossEntropyLoss()

    valid_acc = 0.0
    valid_loss = 0.0
    num_correct = 0.0
    num_samples = 0.0
    Y_true = []
    Y_pred = []
    confidence1 = []
    Y_true_confidence = []
    model.eval()
    for batch_idx, (data, target) in tqdm(enumerate(loaders['validation']), total=len(loaders['validation']),
                                          leave=False):

        if use_cuda:
            data, target = data.to(device, dtype=torch.float), target.to(device)

        output = model(data)

        loss = criterion(output, target)

        sm = torch.nn.Softmax()
        probabilities = sm(output)
        confidence_scores = []
        for c in probabilities:
            c_s = ["{:.5f}".format(i.item()) for i in list(c)]
            confidence_scores.append(c_s)
        _, predictions = output.max(1)

        actual = [t.item() for t in target]
        predicted = [p.item() for p in predictions]
        # confidence = [create_output(scores) for scores in confidence_scores]
        # confidence_actual_class = confidence_scores[actual]
        Y_true += actual
        Y_pred += predicted
        # Y_true_confidence.append(confidence_actual_class)
        confidence1 += confidence_scores
    np.save("confidence_hindi_data_tamil_model128.npy", confidence1)
    np.save("actual_hindi_data_tamil_model128.npy", Y_true)
    accuracy = accuracy_score(Y_true, Y_pred)
    print("Accuracy: ", accuracy)
    print(classification_report(Y_true, Y_pred))
    print(confusion_matrix(Y_true, Y_pred))
    mlflow.log_metric("Accuracy", accuracy)
    return accuracy


with mlflow.start_run(run_id=run_id):
    validation(loaders, model, use_cuda)
# os.remove("validation_data.csv")
