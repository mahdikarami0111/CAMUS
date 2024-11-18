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
    config.enc_arch = [[64, 128, 256, 512, 1024], [128, 256, 512, 1024]]
    config.dec_arch = [[512, 256, 128], [512, 256, 128, 64, 32]]
    config.bn_arch = [512, 128, 512]
    config.image_size = 256
    config.max_epoch = 50
    config.lr = 0.05
    config.train_split = 0.7
    config.test_split = 0.3
    config.val_split = 0
    config.transform = "default"
    return config
