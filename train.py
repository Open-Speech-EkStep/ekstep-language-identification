import torch.nn as nn
import torch.optim as optim

from models.model import get_model
from utils.training import *

torch.manual_seed(0)

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()

train_parameters = load_yaml_file("train_config.yml")["train_parameters"]

# inputs
train_manifest = train_parameters["train_manifest"]

# outputs
current_checkpoint_path = train_parameters["current_checkpoint_path"]
best_checkpoint_path = train_parameters["best_checkpoint_path"]
final_checkpoint_path = train_parameters["final_checkpoint_path"]

# Hyperparameters
batch_size = train_parameters["batch_size"]
learning_rate = train_parameters["learning_rate"]
num_epochs = train_parameters["num_epochs"]
num_workers = train_parameters["num_workers"]

# Load_Data
loaders = load_data_loaders(train_manifest, batch_size, num_workers)

# Load Model
model = get_model(device)

# Display model parameters
show_model_parameters(model)

# Set model hyperparameters
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# start model trainning
trained_model = train(1, num_epochs, device, np.Inf, loaders, model, optimizer, criterion, use_cuda,
                      current_checkpoint_path, best_checkpoint_path, final_checkpoint_path)