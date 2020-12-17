import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from utils.training import *
import sys

tracking_uri = sys.argv[1]
accuracy_score_threshold = sys.argv[2]
mlflow.set_tracking_uri(tracking_uri)
valid_parameters = load_yaml_file("train_config.yml")["train_parameters"]
model_name = valid_parameters["model_name"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

client = MlflowClient()


def load_model_metadata():
    for mv in client.search_model_versions(f"name='{model_name}'"):
        model_version = int(mv.version)


    return mv


run_id = load_model_metadata().run_id
print(f'The run_id is {run_id}')
accuracy_metric = client.get_metric_history(run_id, "test_accuracy")[-1].value
print(f"The test accuracy metric is {accuracy_metric}")
if accuracy_metric >= accuracy_score_threshold:
    print('Model accepted')
else:
    print('Model rejected')
