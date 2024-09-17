from ml_collections import ConfigDict


def get_FCT_config():
    config = ConfigDict()
    config.img_size = 224
    config.data = {
        "root": "data/database_expanded",
        "device": "cuda",
        "type": "N 4CH",
    }
    config.batch_size = 8
    config.num_classes = 1
    config.max_epoch = 10
    config.lr = 0.001
    config.transform = "default"

    return config
