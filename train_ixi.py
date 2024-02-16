"""
Adapted from https://github.com/NVlabs/MUNIT
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer
import argparse
from torch.autograd import Variable
from trainer import MUNIT_Trainer, UNIT_Trainer
import torch.backends.cudnn as cudnn
import torch
import torch.multiprocessing as mp

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
import logging
import platform


if platform.system() == 'Windows':
  sys.path.append(r"E:\我的坚果云\sourcecode\python\util")
else:
  sys.path.append("/home/chenxu/我的坚果云/sourcecode/python/util")
import common_metrics
import common_ixi


def main(logger, opts):
    # Load experiment setting
    config = get_config(opts.config)
    config["gpu"] = opts.gpu
    max_epochs = config['max_epochs']
    display_size = config['display_size']
    config['vgg_model_path'] = opts.output_path

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

    if opts.gpu >= 0:
        trainer.cuda()


    dataset_s = common_ixi.Dataset(opts.data_dir, modality="t2", n_slices=config["input_dim_a"], debug=opts.debug)
    dataset_t = common_ixi.Dataset(opts.data_dir, modality="t1", n_slices=config["input_dim_b"], debug=opts.debug)
    dataloader_s = torch.utils.data.DataLoader(dataset_s, batch_size=opts.batch_size, shuffle=True, pin_memory=True,
                                               drop_last=True)
    dataloader_t = torch.utils.data.DataLoader(dataset_t, batch_size=opts.batch_size, shuffle=True, pin_memory=True,
                                               drop_last=True)
    if opts.do_validation:
        val_data_t, val_data_s = common_ixi.load_test_data(opts.data_dir, "val")

    train_writer = tensorboardX.SummaryWriter(opts.log_dir)
    if not os.path.exists(opts.checkpoint_dir):
        os.makedirs(opts.checkpoint_dir)
    shutil.copy(opts.config, os.path.join(opts.checkpoint_dir, 'config.yaml'))  # copy config file to output folder

    # Start training
    best_psnr = 0
    iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
    for it in range(iterations, max_epochs):
        for batch_id, (data_s, data_t) in enumerate(zip(dataloader_s, dataloader_t)):
            images_a = data_s["image"].to(device)
            images_b = data_t["image"].to(device)

            # Main training code
            trainer.dis_update(images_a, images_b, config)
            trainer.gen_update(images_a, images_b, config)
            if opts.gpu >= 0:
                torch.cuda.synchronize()

        # Dump training stats in log file
        if (it + 1) % config['log_iter'] == 0:
            print("Training Progress: %08d/%08d" % (it + 1, max_epochs))
            write_loss(it, trainer, train_writer)

        if (it + 1) % config['image_display_iter'] == 0:
            msg = "Iter: %d" % (it + 1)

            syn_st, syn_ts = trainer.forward(images_a, images_b)

            train_patch_s_np = images_a.cpu().detach().numpy()
            train_patch_t_np = images_b.cpu().detach().numpy()
            train_syn_st_np = syn_st.cpu().detach().numpy()
            train_syn_ts_np = syn_ts.cpu().detach().numpy()

            gen_images_train = numpy.concatenate([train_patch_s_np, train_syn_st_np, train_syn_ts_np, train_patch_t_np], 3)
            gen_images_train = common_ixi.generate_display_image(gen_images_train, is_seg=False)

            if opts.log_dir:
                try:
                    skimage.io.imsave(os.path.join(opts.log_dir, "gen_images_train.jpg"), gen_images_train)
                except:
                    pass

            if opts.do_validation:
                val_st_psnr = numpy.zeros((val_data_s.shape[0], 1), numpy.float32)
                val_ts_psnr = numpy.zeros((val_data_t.shape[0], 1), numpy.float32)
                val_st_list = []
                val_ts_list = []
                with torch.no_grad():
                    for i in range(val_data_s.shape[0]):
                        val_st = numpy.zeros(val_data_s.shape[1:], numpy.float32)
                        val_ts = numpy.zeros(val_data_t.shape[1:], numpy.float32)
                        used = numpy.zeros(val_data_s.shape[1:], numpy.float32)
                        for j in range(val_data_s.shape[1] - config["input_dim_a"] + 1):
                            val_patch_s = torch.tensor(val_data_s[i:i + 1, j:j + config["input_dim_a"], :, :], device=device)
                            val_patch_t = torch.tensor(val_data_t[i:i + 1, j:j + config["input_dim_b"], :, :], device=device)

                            ret_st, ret_ts = trainer.forward(val_patch_s, val_patch_t)

                            val_st[j:j + config["input_dim_a"], :, :] += ret_st.cpu().detach().numpy()[0]
                            val_ts[j:j + config["input_dim_b"], :, :] += ret_ts.cpu().detach().numpy()[0]
                            used[j:j + config["input_dim_b"], :, :] += 1

                        assert used.min() > 0
                        val_st /= used
                        val_ts /= used

                        st_psnr = common_metrics.psnr(val_st, val_data_t[i])
                        ts_psnr = common_metrics.psnr(val_ts, val_data_s[i])

                        val_st_psnr[i] = st_psnr
                        val_ts_psnr[i] = ts_psnr
                        val_st_list.append(val_st)
                        val_ts_list.append(val_ts)

                msg += "  val_st_psnr:%f/%f  val_ts_psnr:%f/%f" % \
                       (val_st_psnr.mean(), val_st_psnr.std(), val_ts_psnr.mean(), val_ts_psnr.std())
                gen_images_test = numpy.concatenate([val_data_s[0], val_st_list[0], val_ts_list[0], val_data_t[0]], 2)
                gen_images_test = numpy.expand_dims(gen_images_test, 0).astype(numpy.float32)
                gen_images_test = common_ixi.generate_display_image(gen_images_test, is_seg=False)

                if opts.log_dir:
                    try:
                        skimage.io.imsave(os.path.join(opts.log_dir, "gen_images_test.jpg"), gen_images_test)
                    except:
                        pass

                if val_ts_psnr.mean() > best_psnr:
                    best_psnr = val_ts_psnr.mean()

                    if best_psnr > opts.psnr_threshold:
                        trainer.save(opts.checkpoint_dir, "best")

                msg += "  best_ts_psnr:%f" % best_psnr

            logger.info(msg)

        trainer.update_learning_rate()
        trainer.save(opts.checkpoint_dir, "last")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str, default='configs/ixi.yaml', help='Path to the config file.')
    parser.add_argument('--output_path', type=str, default='outputs', help="outputs path")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
    parser.add_argument('--gpu', type=int, default=0, help="gpu device id")
    parser.add_argument('--batch_size', type=int, default=16, help="batch size")
    parser.add_argument('--data_dir', type=str, default=r'data', help='path of the dataset')
    parser.add_argument('--log_dir', type=str, default=r'logs', help="log file dir")
    parser.add_argument('--checkpoint_dir', type=str, default=r'checkpoints', help="checkpoint file dir")
    parser.add_argument('--pretrained_tag', type=str, default='final', choices=['best','final'], help="pretrained file tag")
    parser.add_argument('--logfile', type=str, default='', help="log file")
    parser.add_argument('--aug_sigma', type=float, default=0.5, help="augmentation parameter:sigma")
    parser.add_argument('--aug_points', type=float, default=20, help="augmentation parameter:points")
    parser.add_argument('--aug_rotate', type=float, default=0, help="augmentation parameter:rotate")
    parser.add_argument('--aug_zoom', type=float, default=1., help="augmentation parameter:zoom")
    parser.add_argument('--psnr_threshold', type=float, default=18.0, help="only save the checkpoint file when PSNR reach this threshold")
    parser.add_argument('--do_validation', type=int, default=1, help="whether do validation during training")
    parser.add_argument('--debug', type=int, default=0, help="debug flag")

    opts = parser.parse_args()

    # 日记信息
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
    logc = logging.StreamHandler()
    logc.setFormatter(formatter)
    logger.addHandler(logc)
    if len(opts.logfile) > 0:
        logf = logging.FileHandler(opts.logfile)
        logf.setFormatter(formatter)
        logger.addHandler(logf)

    main(logger, opts)

