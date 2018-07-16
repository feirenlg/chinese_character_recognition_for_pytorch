import warnings


class Config(object):
    DEBUG_FILE = './tmp/debug'

    ENVIRONMENT = 'default'

    TRAIN_DATA_ROOT = 'F:/HWDB1/data/train'

    TEST_DATA_ROOT = 'F:/HWDB1/data/test'

    CHECKPOINTS_ROOT = './checkpoints/'

    LOAD_MODEL_PATH = None

    BATCH_SIZE = 128

    USE_GPU = True

    NUM_WORKERS = 4

    PRINT_FREQ = 20

    MAX_EPOCH = 20

    LEARNING_RATE = 0.001

    MOMENTUM_RATE = 0.9

    DROPOUT_RATE = 0.5

    WEIGHT_DECAY = 0.0005

    MAX_CLASSES = 3755

    RE_SIZE = 56

    INPUT_SIZE = 64

    PADDING_SIZE = 4

    MEAN_R = 0.674

    MEAN_G = 0.674

    MEAN_B = 0.674

    STD_R = 0.415

    STD_G = 0.415

    STD_B = 0.415

    LEARNING_RATE_DECAY = 0.5

    EPOCH_DECAY = 3

    FIRST_OUT_CHANNEL = 64



def parse(self, kwargs):
    for key, val in kwargs.items():
        if not hasattr(self, key):
            warnings.warn("Warning: opt has not attribute %s" % key)
        setattr(self, key, val)

    print('user config:')
    for key, val in self.__class__.__dict__.items():
        if not key.startswith('__'):
            print(key, getattr(self, key))


Config.parse = parse
conf = Config()
