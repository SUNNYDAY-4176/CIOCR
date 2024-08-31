from libs.dataset.data import ROOT, build_dataset, multibatch_collate_fn
from libs.dataset.transform import TestTransform
from libs.utils.logger import AverageMeter
from libs.utils.utility import parse_args, write_mask
from libs.utils.utility import parse_args, write_mask, save_checkpoint
from libs.models.models import STAN
import torch
import torch.utils.data as data
import libs.utils.logger as logger
import os
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
MAX_FLT = 1e6
opt, _ = parse_args()
device = 'cuda:{}'.format(opt.gpu_id)
use_gpu = torch.cuda.is_available() and int(opt.gpu_id) >= 0
logger.setup(filename='test_out.log', resume=False)
log = logger.getLogger(__name__)


def main(model_name='', model_path=''):

    # Data
    log.info('Preparing dataset %s' % opt.valset)

    input_dim = opt.input_size

    test_transformer = TestTransform(size=input_dim)
    testset = build_dataset(
        name=opt.valset,
        train=False, 
        transform=test_transformer, 
        samples_per_video=1
        )

    testloader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=opt.workers,
                                collate_fn=multibatch_collate_fn)
    log.info("Creating model")
    net = STAN(opt)
    log.info('Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))
    net.eval()
    if use_gpu:
        net.to(device)
    for p in net.parameters():
        p.requires_grad = False

    log.info('Loading weights from checkpoint {}'.format(model_path))
    assert os.path.isfile(model_path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(model_path, map_location=device)
    try:
        net.load_param(checkpoint['state_dict'])
    except:
        net.load_param(checkpoint)
    log.info('Runing model on dataset {}, totally {:d} videos'.format(opt.valset, len(testloader)))
    test_adaptive_memory(
                        testloader,
                        model=net,
                        use_cuda=use_gpu,
                        opt=opt,
                        model_name=model_name)
    log.info('Results are saved at: {}'.format(os.path.join(ROOT, opt.output_dir, opt.valset)))


def test_adaptive_memory(testloader, model, use_cuda, model_name, opt):
    data_time = AverageMeter()
    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):

            frames, masks, objs, infos = data

            if use_cuda:
                frames = frames.squeeze(0).to(device)
                masks = masks.squeeze(0).to(device)

            load_number = int(frames.size(0) / opt.sampled_frames) if (frames.size(
                0) % opt.sampled_frames) == 0 else int((frames.size(0) / opt.sampled_frames) + 1)
            num_objects, info, max_obj = objs[0], infos[0], (masks.shape[1] - 1)

            # t1 = time.time()

            T, _, H, W = frames.shape
            pred, keys, vals, scales, keys_dict, vals_dict = [], [], [], [], {}, {}

            for C in range(1, load_number):
                Clip_first_idx, Clip_last_idx = (C - 1) * opt.sampled_frames, (C) * opt.sampled_frames
                Clip_Frame, Clip_Mask = frames[Clip_first_idx:Clip_last_idx, :, :, :], masks[
                                                                                       Clip_first_idx:Clip_last_idx, :,
                                                                                       :, :]
                clip_key, clip_val, r4 = model(frame=Clip_Frame, mask=Clip_Mask, num_objects=num_objects)
                keys.append(clip_key)
                vals.append(clip_val)
                keys_dict[C] = clip_key
                vals_dict[C] = clip_val

                tmp_key, tmp_val = torch.cat(keys, dim=0), torch.cat(vals, dim=0)
                logit_list, _ = model(frame=Clip_Frame,
                                                                                               keys=tmp_key,
                                                                                               values=tmp_val,
                                                                                               num_objects=num_objects,
                                                                                               max_obj=max_obj,
                                                                                               opt=opt,
                                                                                               Clip_idx=C,
                                                                                               keys_dict=keys_dict,
                                                                                               vals_dict=vals_dict,
                                                                                               patch=2)
                for l in range(len(logit_list)):
                    logit = logit_list[l]
                    out = torch.softmax(logit, dim=1)
                    pred.append(out)

            if (frames.size(0) - Clip_last_idx) > 0:
                Clip_Frame, Clip_Mask = frames[-int(opt.sampled_frames):, :, :, :], masks[-int(opt.sampled_frames):, :, :, :]
                clip_key, clip_val, r4 = model(frame=Clip_Frame, mask=Clip_Mask, num_objects=num_objects)
                keys.append(clip_key)
                vals.append(clip_val)
                keys_dict[C + 1] = clip_key
                vals_dict[C + 1] = clip_val

                tmp_key, tmp_val = torch.cat(keys, dim=0), torch.cat(vals, dim=0)
                logit_list, _ = model(frame=Clip_Frame,
                                                                                               keys=tmp_key,
                                                                                               values=tmp_val,
                                                                                               num_objects=num_objects,
                                                                                               max_obj=max_obj,
                                                                                               opt=opt,
                                                                                               Clip_idx=C + 1,
                                                                                               keys_dict=keys_dict,
                                                                                               vals_dict=vals_dict,
                                                                                               patch=2)

                if frames.size(0) % opt.sampled_frames != 0:
                    logit_list = logit_list[-1 * int(frames.size(0) % opt.sampled_frames):]

                for l in range(len(logit_list)):
                    logit = logit_list[l]
                    out = torch.softmax(logit, dim=1)
                    pred.append(out)

            pred = torch.cat(pred, dim=0)
            pred = pred.detach().cpu().numpy()
            assert num_objects == 1
            write_mask(pred, info, opt, directory=opt.output_dir, model_name='{}'.format(model_name))
    return data_time.sum


if __name__ == '__main__':

    models_path = './ckpt/'
    models = [file for file in os.listdir(models_path) if file.endswith('pth.tar')]
    for model_name in models:
        model_path = os.path.join(models_path, model_name)
        model_name = model_name.replace('.', '_')
        main(model_name=model_name, model_path=model_path)
