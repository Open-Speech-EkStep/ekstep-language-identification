import os

import numpy as np
import torch
import yaml

from utils import utils

# check cuda available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.manual_seed(0)


def load_model(model_path):
    # model = get_model(device)
    if os.path.isfile(model_path):
        model = torch.load(model_path, map_location=device)
        # model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        print("Model loaded from ", model_path)
    else:
        print("Saved model not found")
        exit(1)
    return model


def forward(audio, model, mode='test'):
    try:
        model.eval()
        spec = utils.load_data(audio, mode=mode)[np.newaxis, ...]
        feats = np.asarray(spec)
        feats = torch.from_numpy(feats)
        feats = feats.unsqueeze(0)
        feats = feats.to(device)
        label = model(feats.float())
        return label
    except:
        print("File error ", audio)


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


def evaluation(audio_path, model_path):
    model = load_model(model_path)
    model_output = forward(audio_path, model=model)
    # _, prediction = model_output.max(1)
    sm = torch.nn.Softmax()
    probabilities = sm(model_output)
    confidence_scores = ["{:.5f}".format(i.item()) for i in list(probabilities[0])]
    return create_output(confidence_scores)


# if __name__ == "__main__":
#     model_path = './checkpoints/best_model.pt'
#     print(evaluation('/Users/ps/language_identification/Audios/024-M-58_126.wav', model_path))
