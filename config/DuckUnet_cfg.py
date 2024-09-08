from ml_collections import ConfigDict


def get_DuckNet_config():
    config = ConfigDict()
    config.in_channels = 1
    config.out_channels = 1
    config.num_classes = 1
    config.depth = 5
    config.init_features = 32
    config.normalization = "batch"
    config.interpolation = "nearest"
    config.out_activation = "sigmoid"
    config.use_multiplier = False
    return config

def get_DuckNet_train_config():
    config = ConfigDict()
    config.data = {
        "root": "data/database_expanded",
        "device": "cuda",
        "type": "N 4CH",
    }
    config.batch_size = 2
    config.num_classes = 1
    config.image_size = 224
    config.max_epoch = 10
    config.lr = 0.001
    config.transform = "default"
    return config

