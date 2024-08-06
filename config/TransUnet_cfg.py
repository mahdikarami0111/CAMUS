from ml_collections import ConfigDict


def get_TransUnet_config():
    config = ConfigDict()
    config.patches = ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.classifier = 'seg'
    config.representation_size = None
    config.patch_size = 16

    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 1
    config.activation = 'softmax'

    config.patches.grid = (14, 14)
    config.resnet = ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.skip_channels = [512, 256, 64, 16]
    config.n_skip = 3
    return config


def get_train_config():
    config = ConfigDict()
    config.data = {
        "root": "data/database_expanded",
        "device": "cuda",
        "type": "N 4CH",
    }
    config.batch_size = 8
    config.num_classes = 1
    config.image_size = 224
    config.max_epoch = 10
    config.lr = 0.05
    config.train_split = 0.7
    config.test_split = 0.3
    config.val_split = 0
    config.transform = "default"
    return config
