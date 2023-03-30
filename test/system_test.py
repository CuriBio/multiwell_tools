import json
from src.multi_well_video import well_signals


if __name__ == '__main__':
    with open("./test/config.json") as json_file:
        setup_config = json.load(json_file)
        well_signals(setup_config=setup_config)
