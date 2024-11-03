from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()
    config.data = {
        "root": "data/database_expanded",
        "device": "cuda",
        "type": "N 4CH",
    }
    config.batch_size = 2
    config.num_classes = 1
    config.arch = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 1024]
    config.image_size = 256
    config.max_epoch = 50
    config.lr = 0.05
    config.train_split = 0.7
    config.test_split = 0.3
    config.val_split = 0
    config.transform = "default"
    return config
