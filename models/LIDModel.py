import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from tqdm import tqdm

from MlLifeCycle.BasePytorchModelArchitecture import BaseModelArchitecture, MlLifeCycle
from loaders.data_loader import SpeechDataGenerator


class LIDModel(BaseModelArchitecture):
    def __init__(self, config_path=None):
        super().__init__(config_path)

        self.config = None
        self.device = None
        self.model = None
        self.task = None
        self.loader = None
        self.optimizer = None
        self.criterion = None
        self.valid_loss_min = 0.0

        self.config = self.load_config()["train_parameters"]
        self.config["batch_size"] = int(self.config["batch_size"])
        self.config["learning_rate"] = float(self.config["learning_rate"])
        self.config["num_epochs"] = int(self.config["num_epochs"])
        self.config["num_workers"] = int(self.config["num_workers"])
        self.config["num_classes"] = int(self.config["num_classes"])
        self.config["score_threshold"] = float(self.config["score_threshold"])
        self.logger_client = MlLifeCycle(tracking_uri=self.config["tracking_uri"],
                                         experiment_name=self.config["experiment_name"])

    def load_config(self) -> dict:
        """Placeholder for model config loader, returns model parameters as dict"""
        read_dict = {}
        with open(self.config_path, 'r') as file:
            read_dict = yaml.safe_load(file)
        return read_dict

    def restore_model_training(self, model_name=None):
        if not model_name:
            model_name = self.config["model_name"] + "_last"
        models = self.logger_client.fetch_models(model_name=model_name)
        if models:
            latest_model_version = list(models.keys())[-1]
            latest_model_uri = models[latest_model_version]["model_uri"]
            state = self.logger_client.load_model(model_uri=latest_model_uri)
            run_id = models[latest_model_version]["model_run_id"]
            return state, run_id
        else:
            print("No model found")
            pass

    def iterator(self):
        assert (self.task == "train" or self.task == "valid" or self.task == "test"), \
            "Invalid task, can only be one of (train, valid, test)."
        if self.task == "train":
            self.model.train()
        elif self.task == "valid" or self.task == "test":
            self.model.eval()
        loss = np.inf
        predictions = []
        targets = []
        for batch_idx, (data, target) in tqdm(enumerate(self.loader[self.task]), total=len(self.loader[self.task]),
                                              leave=False):
            data, target = data.to(self.device, dtype=torch.float), target.to(self.device)
            if self.task == "train":
                self.optimizer.zero_grad()

            output = self.model(data.float())

            loss = self.criterion(output, target)
            if self.task == "train":
                loss.backward()
                self.optimizer.step()
            loss = loss + ((1 / (batch_idx + 1)) * (loss.data - loss))
            if self.task == "train" or self.task == "valid":
                self.logger_client.log_metric(metrics={self.task + "_batch_Loss": loss})

            _, predictions = output.max(1)
            temp_predict = [pred.item() for pred in predictions]
            temp_target = [actual.item() for actual in target]
            predictions = predictions + temp_predict
            targets = targets + temp_target

        loss = loss / len(self.loader[self.task].dataset)
        predictions = [p.item() for p in predictions]
        targets = [t.item() for t in targets]
        results = {"Loss": loss, "Predictions": predictions, "Targets": targets}

        return results

    def get_optimizer(self):
        assert self.model, "Model set to None, Set model in set_train_parameters"
        return optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])

    @staticmethod
    def get_criterion():
        return nn.CrossEntropyLoss()

    def get_device(self):
        if self.config["device"].lower() == "cpu":
            return torch.device("cpu")
        else:
            return torch.device("cuda")

    def get_loader(self, manifest=None):
        dataset = SpeechDataGenerator(manifest=manifest, mode='train')
        loader = DataLoader(dataset=dataset, batch_size=self.config["batch_size"], shuffle=True,
                            num_workers=self.config["num_workers"])
        return loader

    def build_train_data_loaders(self):
        loaders = {"train": self.get_loader(self.config["train_manifest"]),
                   "valid": self.get_loader(self.config["valid_manifest"])}
        return loaders

    def get_state(self, epoch):
        state = {
            'epoch': epoch + 1,
            'valid_loss_min': self.valid_loss_min,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        return state

    def build_model(self):
        assert (self.config and self.device), "device set to None,Set pytorch device in set_train_parameters"
        model = resnet18(pretrained=self.config["pretrained"])
        model.fc = nn.Linear(512, self.config["num_classes"])
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.to(self.device, dtype=torch.float)
        return model

    @staticmethod
    def cal_accuracy(predictions, targets):
        return accuracy_score(targets, predictions)

    def set_train_parameters(self):
        """Place holder for setting train parameters like model, loaders, optimizer, criterion, device"""
        self.device = self.get_device()
        self.model = self.build_model()
        self.task = None
        self.loader = self.build_train_data_loaders()
        self.optimizer = self.get_optimizer()
        self.criterion = self.get_criterion()

    def train(self):
        self.set_train_parameters()
        final_results = {}
        start_epochs = 0
        run_id = None
        if self.config["restore_training"]:
            state, run_id = self.restore_model_training()
            self.model.load_state_dict(state['state_dict'])
            self.optimizer.load_state_dict(state['optimizer'])
            start_epochs = state['epoch']
            self.valid_loss_min = state['valid_loss_min']
        with mlflow.start_run(run_id=run_id, run_name=self.config["run_name"]):
            self.logger_client.log_param(self.config)
            for epoch in range(start_epochs, self.config["num_epochs"] + 1):
                self.task = "train"
                train_result = self.iterator()
                self.task = "valid"
                valid_result = self.iterator()
                final_results["train_loss"] = train_result["Loss"]
                final_results["valid_loss"] = valid_result["Loss"]
                final_results["train_accuracy"] = self.cal_accuracy(train_result["Predictions"],
                                                                    train_result["Targets"])
                final_results["valid_accuracy"] = self.cal_accuracy(valid_result["Predictions"],
                                                                    valid_result["Targets"])
                self.logger_client.log_metric(metrics=final_results)

                if valid_result["Loss"] < self.valid_loss_min:
                    self.valid_loss_min = valid_result["Loss"]
                    self.logger_client.log_model(model=self.model, model_name=self.config["model_name"] + "_best")

                state = self.get_state(epoch)
                self.logger_client.log_model(model=state, model_name=self.config["model_name"] + "_last")

    def test(self):
        self.set_train_parameters()
        final_results = {}
        # override arguments for test
        model_name = self.config["model_name"] + "_best"
        self.model, run_id = self.restore_model_training(model_name=model_name)
        if self.model and run_id:
            self.loader = self.get_loader(manifest=self.config["test_manifest"])
            self.task = "test"
            with mlflow.start_run(run_id=run_id, run_name=self.config["run_name"]):
                self.logger_client.log_param(self.config)
                results = self.iterator()
                final_results["test_loss"] = results["Loss"]
                final_results["test_accuracy"] = self.cal_accuracy(results["Predictions"], results["Targets"])
                self.logger_client.log_metric(metrics=final_results)
        else:
            print(f"No best model found with name: {model_name}, exiting...")
            exit()
