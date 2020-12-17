import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from utils.training import *
import sys

accuracy_score_threshold = sys.argv[1]
valid_parameters = load_yaml_file("train_config.yml")["train_parameters"]
model_name = valid_parameters["model_name"]

client = MlflowClient()


def load_model():
    for mv in client.search_model_versions(f"name='{model_name}'"):
        model_version = int(mv.version)

    model = mlflow.pytorch.load_model(
        model_uri=f"models:/{model_name}/{model_version}"
    )
    print(f"Model fetched with name : {model_name} and version {model_version}")
    print(model)
    return model


model = load_model()

run_id = model.metadata.run_id
accuracy_metric = client.get_metric_history(run_id, "Accuracy")[-1].value
print(f"The accuracy metric is {accuracy_metric}")
if accuracy_metric >= accuracy_score_threshold:
    print('Model accepted')
else:
    print('Model rejected')
