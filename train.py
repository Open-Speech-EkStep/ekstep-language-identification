import torch.nn as nn
import torch.optim as optim

from models.model import get_model
from utils.training import *
import mlflow
# torch.manual_seed(0)
mlflow.set_tracking_uri('http://0.0.0.0:9000')
# Set Device
with mlflow.start_run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = torch.cuda.is_available()

    train_parameters = load_yaml_file("train_config.yml")["train_parameters"]

    # inputs
    train_manifest = train_parameters["train_manifest"]
    valid_manifest = train_parameters["valid_manifest"]
    # outputs
    checkpoint_path = train_parameters["checkpoint_path"]

    # Hyperparameters
    batch_size = int(train_parameters["batch_size"])
    learning_rate = float(train_parameters["learning_rate"])
    num_epochs = int(train_parameters["num_epochs"])
    num_workers = int(train_parameters["num_workers"])
    num_classes = int(train_parameters["num_classes"])

    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("num_workers", num_workers)
    mlflow.log_param("num_classes", num_classes)

    # Load_Data
    loaders = load_data_loaders(train_manifest, valid_manifest, batch_size, num_workers)

    # Load Model
    model = get_model(device, num_classes, pretrained=False)

    # Display model parameters
    show_model_parameters(model)

    # Set model hyperparameters
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # start model training
    trained_model = train(1, num_epochs, device, np.Inf, loaders, model, optimizer, criterion, use_cuda, checkpoint_path,
                          save_for_each_epoch=True)
    mlflow.pytorch.log_model(trained_model, "LID_MODEL", registered_model_name="LID_MODEL")
