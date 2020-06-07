from utils.regression_trainer import RegTrainer
import argparse
import os
import torch
args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train ')
    parser.add_argument('--data-dir', default='./processedData',
                        help='training data directory')
    parser.add_argument('--save-dir', default='./logs',
                        help='directory to save models.')

    parser.add_argument('--lr', type=float, default=1e-5,
                        help='the initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='the weight decay')
    parser.add_argument('--resume', default='',
                        help='the path of resume training model')
    parser.add_argument('--max-model-num', type=int, default=1,
                        help='max models num to save ')
    parser.add_argument('--max-epoch', type=int, default=3000,
                        help='max training epoch')
    parser.add_argument('--use-amp', type=bool, default=False,
                        help='use amp')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='train batch size')
    parser.add_argument('--num-gpus', default=10, help='assign device')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='the num of training process')

    parser.add_argument('--is-gray', type=bool, default=False,
                        help='whether the input image is gray')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='the crop size of the train image')
    parser.add_argument('--downsample-ratio', type=int, default=8,
                        help='downsample ratio')

    parser.add_argument('--use-background', type=bool, default=True,
                        help='whether to use background modelling')
    parser.add_argument('--sigma', type=float, default=8.0,
                        help='sigma for likelihood')
    parser.add_argument('--background-ratio', type=float, default=0.15,
                        help='background ratio')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning import Trainer
    from pytorch_lightning.logging import TestTubeLogger

    args = parse_args()
    system = RegTrainer(args)
    checkpoint_callback = ModelCheckpoint(filepath=os.path.join(f'{args.save_dir}/ckpts',
                                                                '{epoch:02d}'),
                                          monitor='val/mean_val_mse',
                                          mode='min',
                                          save_top_k=5, )

    logger = TestTubeLogger(
        save_dir=args.save_dir,
        name=args.save_dir,
        debug=False,
        create_git_tag=False
    )

    trainer = Trainer(max_epochs=args.max_epoch,
                      checkpoint_callback=checkpoint_callback,
                      logger=logger,
                      early_stop_callback=None,
                      weights_summary=None,
                      progress_bar_refresh_rate=1,
                      gpus=args.num_gpus,
                      distributed_backend='ddp' if args.num_gpus > 1 else None,
                      num_sanity_val_steps=0 if args.num_gpus > 1 else 5,
                      benchmark=True,
                      precision=16 if args.use_amp else 32,
                      amp_level='O1',
                      val_check_interval = 0.1)

    trainer.fit(system)

