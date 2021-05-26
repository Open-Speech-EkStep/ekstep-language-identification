import sys

from models.LIDModel import LIDModel

config_path = sys.argv[0]
lid_model = LIDModel(config_path=config_path + "/train_config.yml")

lid_model.train()
