import mlflow
from mlflow.tracking import MlflowClient


class MlflowLifeCycle:
    def __init__(self, tracking_uri, experiment_name):
        assert tracking_uri, "Empty tracking url, Please provide url of MlLifeCycle server."
        mlflow.set_tracking_uri(tracking_uri)
        self.tracking_uri = tracking_uri
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        if experiment_name:
            mlflow.set_experiment(experiment_name)
        else:
            print("Experiment name not found, result will be tracked under Default.")

    @staticmethod
    def log_metric(metrics: dict):
        # for metric in metrics:
        #     assert float(metrics[metric]), "Metric value value can only be int or float."

        for metric in metrics:
            mlflow.log_metric(metric, float(metrics[metric]))

    @staticmethod
    def log_param(parameters: dict):
        for param in parameters:
            mlflow.log_param(param, parameters[param])

    def fetch_models(self, model_name: str) -> dict:
        model_details = {}
        for mv in self.client.search_model_versions(f"name='{model_name}'"):
            model_version = int(mv.version)
            model_run_id = mv.run_id
            model_uri = self.client.get_model_version_download_uri(model_name, model_version)
            model_details[model_version] = {"model_uri": model_uri, "model_run_id": model_run_id}

        return model_details

    def log_model(self, model, model_name):
        """Placeholder for model logging at that depends on framework used"""
        pass

    def load_model(self, model_uri):
        """Placeholder for model loading as that depends on framework used"""
        pass

    def model_transition(self, model_name=None, model_version=None, stage=None):
        assert model_name, "Please provide model name"
        assert model_version, "Please provide model version"
        assert stage, "Please provide model stage"
        self.client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage=stage
        )
        print(f"Promoted model to {stage} layer")

