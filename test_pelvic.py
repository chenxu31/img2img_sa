"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer
import argparse
from torch.autograd import Variable
from trainer import MUNIT_Trainer, UNIT_Trainer
import torch.backends.cudnn as cudnn
import torch
import logging
import skimage.io
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import sys
import tensorboardX
import shutil
import numpy
import pdb
from skimage.metrics import structural_similarity as SSIM


sys.path.append(r"E:\我的坚果云\sourcecode\python\util")
import common_metrics
import common_pelvic_pt as common_pelvic

def main(logger, opts):
    # Load experiment setting
    config = get_config(opts.config)
    config["gpu"] = opts.gpu
    config['vgg_model_path'] = opts.output_dir

    if not os.path.exists(opts.output_dir):
        os.makedirs(opts.output_dir)

    if opts.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(opts.gpu)
        device = torch.device("cuda")
        cudnn.benchmark = True
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        device = torch.device("cpu")

    # Setup model and data loader
    if opts.trainer == 'MUNIT':
        trainer = MUNIT_Trainer(config)
    elif opts.trainer == 'UNIT':
        trainer = UNIT_Trainer(config)
    else:
        sys.exit("Only support MUNIT|UNIT")

    state_dict = torch.load(os.path.join(opts.checkpoint_dir, 'gen_%s.pt' % opts.pretrained_tag))
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])

    if opts.gpu >= 0:
        trainer.cuda()

    test_ids_t = common_pelvic.load_data_ids(opts.data_dir, "testing", "treat")
    test_data_s, test_data_t, _, _ = common_pelvic.load_test_data_ex(opts.data_dir, valid=opts.valid)

    test_st_psnr = numpy.zeros((len(test_data_s),), numpy.float32)
    test_ts_psnr = numpy.zeros((len(test_data_t),), numpy.float32)
    test_st_ssim = numpy.zeros((len(test_data_s),), numpy.float32)
    test_ts_ssim = numpy.zeros((len(test_data_t),), numpy.float32)
    test_st_mae = numpy.zeros((len(test_data_s),), numpy.float32)
    test_ts_mae = numpy.zeros((len(test_data_t),), numpy.float32)
    test_st_list = []
    test_ts_list = []
    msg_detail = ""
    with torch.no_grad():
        for i in range(len(test_data_s)):
            test_st = numpy.zeros(test_data_s[i].shape, numpy.float32)
            test_ts = numpy.zeros(test_data_t[i].shape, numpy.float32)
            used = numpy.zeros(test_data_s[i].shape, numpy.float32)
            for j in range(test_data_s[i].shape[0] - config["input_dim_a"] + 1):
                test_patch_s = torch.tensor(numpy.expand_dims(test_data_s[i][j:j + config["input_dim_a"], :, :], 0), device=device)
                test_patch_t = torch.tensor(numpy.expand_dims(test_data_t[i][j:j + config["input_dim_b"], :, :], 0), device=device)

                ret_st, ret_ts = trainer.forward(test_patch_s, test_patch_t)

                test_st[j:j + config["input_dim_a"], :, :] += ret_st.cpu().detach().numpy()[0]
                test_ts[j:j + config["input_dim_b"], :, :] += ret_ts.cpu().detach().numpy()[0]
                used[j:j + config["input_dim_b"], :, :] += 1

            assert used.min() > 0
            test_st /= used
            test_ts /= used

            """
            if opts.output_dir:
                common_pelvic.save_nii(test_ts, os.path.join(opts.output_dir, "syn_%s.nii.gz" % test_ids_t[i]))
            """

            st_psnr = common_metrics.psnr(test_st, test_data_t[i])
            ts_psnr = common_metrics.psnr(test_ts, test_data_s[i])
            st_ssim = SSIM(test_st, test_data_t[i])
            ts_ssim = SSIM(test_ts, test_data_s[i])
            st_mae = abs(common_pelvic.restore_hu(test_st) - common_pelvic.restore_hu(test_data_t[i])).mean()
            ts_mae = abs(common_pelvic.restore_hu(test_ts) - common_pelvic.restore_hu(test_data_s[i])).mean()

            test_st_psnr[i] = st_psnr
            test_ts_psnr[i] = ts_psnr
            test_st_ssim[i] = st_ssim
            test_ts_ssim[i] = ts_ssim
            test_st_mae[i] = st_mae
            test_ts_mae[i] = ts_mae
            test_st_list.append(test_st)
            test_ts_list.append(test_ts)

            msg_detail += "  %s_psnr: %f  %s_ssim: %f\n" % (test_ids_t[i], ts_psnr, test_ids_t[i], ts_ssim)

    msg = "test_st_psnr:%f/%f  test_st_ssim:%f/%f  test_ts_psnr:%f/%f  test_ts_ssim:%f/%f  test_ts_mae:%f/%f  test_ts_mae:%f/%f" % \
          (test_st_psnr.mean(), test_st_psnr.std(), test_st_ssim.mean(), test_st_ssim.std(),
           test_ts_psnr.mean(), test_ts_psnr.std(), test_ts_ssim.mean(), test_ts_ssim.std(),
           test_ts_mae.mean(), test_ts_mae.std(), test_ts_mae.mean(), test_ts_mae.std())
    logger.info(msg)
    logger.info(msg_detail)

    if opts.output_dir:
        with open(os.path.join(opts.output_dir, "result.txt"), "w") as f:
            f.write(msg + msg_detail)

        numpy.save(os.path.join(opts.output_dir, "st_psnr.npy"), test_st_psnr)
        numpy.save(os.path.join(opts.output_dir, "ts_psnr.npy"), test_ts_psnr)
        numpy.save(os.path.join(opts.output_dir, "st_ssim.npy"), test_st_ssim)
        numpy.save(os.path.join(opts.output_dir, "ts_ssim.npy"), test_ts_ssim)
        numpy.save(os.path.join(opts.output_dir, "st_mae.npy"), test_st_mae)
        numpy.save(os.path.join(opts.output_dir, "ts_mae.npy"), test_ts_mae)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str, default='configs/pelvic.yaml', help='Path to the config file.')
    parser.add_argument('--output_dir', type=str, default='outputs', help="outputs path")
    parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
    parser.add_argument('--gpu', type=int, default=0, help="gpu device id")
    parser.add_argument('--data_dir', type=str, default=r'data', help='path of the dataset')
    parser.add_argument('--checkpoint_dir', type=str, default=r'checkpoints', help="checkpoint file dir")
    parser.add_argument('--pretrained_tag', type=str, default='best', choices=['best','final'], help="pretrained file tag")
    parser.add_argument('--valid', type=int, default=1, help="xxx")

    opts = parser.parse_args()

    # 日记信息
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
    logc = logging.StreamHandler()
    logc.setFormatter(formatter)
    logger.addHandler(logc)

    main(logger, opts)

