import mlflow

from MlLifeCycle.MlflowLifeCycle import MlflowLifeCycle


class MlLifeCycle(MlflowLifeCycle):
    def __init__(self, tracking_uri=None, experiment_name=None):
        super().__init__(tracking_uri, experiment_name)

    def log_model(self, model, model_name):
        """Placeholder for model logging at that depends on framework used"""
        mlflow.pytorch.log_model(model, model_name, registered_model_name=model_name, await_registration_for=1)

    def load_model(self, model_uri):
        """Placeholder for model loading as that depends on framework used"""
        return mlflow.pytorch.load_model(model_uri)

    def log_artifact(self, file_name):
        mlflow.log_artifact(file_name)


class BaseModelArchitecture:
    def __init__(self, config_path=None):

        if config_path:
            self.config_path = config_path
        else:
            print("Please enter config path")
            exit()

    def load_config(self) -> dict:
        """Placeholder for model config loader, returns model parameters as dict"""
        pass

    def restore_model_training(self, model_name=None):
        """Place holder for method to restore model training"""
        pass

    def iterator(self):
        """Place holder for inner iterator on batches method"""
        pass

    def get_optimizer(self):
        """Place holder for setting optimizer"""
        pass

    @staticmethod
    def get_criterion():
        """Place holder for setting criterion"""
        pass

    def get_device(self):
        """Place holder for getting device for task"""
        pass

    def get_loader(self, manifest=None):
        pass

    def build_train_data_loaders(self):
        """Place holder for creating data loaders"""
        pass

    def build_model(self):
        """Place holder for building model"""
        pass

    def set_train_parameters(self):
        """Place holder for setting train parameters like model, loaders, optimizer, criterion, device"""
        pass

    def train(self):
        """Place holder for train method"""
        pass

    def test(self):
        """Place holder for test method"""
        pass
