from ml_collections import ConfigDict


def get_train_cfg():
    config = ConfigDict()
    config.data = {
        "root": "data/database_expanded",
        "device": "cuda",
        "type": "N 4CH",
    }

    config.batch_size = 2
    config.num_classes = 2
    config.image_size = 224
    config.max_epoch = 20
    config.lr = 0.00015
    config.train_split = 0.7
    config.test_split = 0.1
    config.val_split = 0.2
    config.transform = "default"
    config.bilinear = False
    return config
