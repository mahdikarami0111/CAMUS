from ml_collections import ConfigDict


def get_TransAttUnet_train_config():
    config = ConfigDict()
    config.data = {
        "root": "data/database_expanded",
        "device": "cuda",
        "type": "N 4CH",
    }
    config.batch_size = 4
    config.num_classes = 1
    config.image_size = 224
    config.max_epoch = 10
    config.lr = 0.001
    config.transform = "default"

    return config
