from yacs.config import CfgNode

OPTION = CfgNode()
OPTION.valset = 'VOS'
OPTION.input_size = (240, 240)
OPTION.sampled_frames = 3
OPTION.data_backend = 'PIL'
OPTION.keydim = 128
OPTION.valdim = 512
OPTION.arch = 'resnet50'
OPTION.epoch_per_test = 1
OPTION.correction_rate = 150
OPTION.correction_momentum = 0.9
OPTION.loop = 10
OPTION.adapt_memory = True
OPTION.memory_max_Clip = 5
OPTION.spatial_decay = 0.92
OPTION.temporal_decay = 0.6
OPTION.checkpoint = 'ckpt'
OPTION.initial = ''
OPTION.resume = ''
OPTION.video_path = ''
OPTION.mask_path = ''
OPTION.gpu_id = 0
OPTION.workers = 1
OPTION.save_indexed_format = 'index'
OPTION.output_dir = 'output'


def sanity_check(opt):
    assert isinstance(opt.valset, str), \
        'validation set should be a single dataset'
    assert opt.arch in ['resnet18', 'resnet34', 'resnet50', 'resnet101']


def getCfg():

    return OPTION.clone()