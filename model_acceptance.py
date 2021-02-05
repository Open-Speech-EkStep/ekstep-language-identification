from models.LIDModel import LIDModel

lid_model = LIDModel(config_path="train_config.yml")

score_threshold = lid_model.config["score_threshold"]
model_name = lid_model.config["model_name"] + "_best"

model_details = lid_model.logger_client.fetch_models(model_name=model_name)
best_model_version = list(model_details.keys())[-1]
best_model_run_id = model_details[best_model_version]["model_run_id"]
score = lid_model.logger_client.client.get_metric_history(run_id=best_model_run_id, key="test_accuracy")[-1].value

print(f"The test accuracy metric is {score}")

if score >= float(score_threshold):
    print('Model accepted')
    lid_model.logger_client.model_transition(model_name=model_name, model_version=best_model_version, stage='staging')
else:
    print('Model rejected')
