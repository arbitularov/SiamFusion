from __future__ import absolute_import, print_function

import os
import sys
import torch
from torch.utils.data import DataLoader

from got10k.datasets import ImageNetVID, GOT10k
from pairwise import Pairwise
from siamfc import TrackerSiamFC
from got10k.experiments import *

from config import config

if __name__ == '__main__':

    # setup dataset
    name = 'GOT-10k'
    assert name in ['VID', 'GOT-10k', 'All']
    if name == 'GOT-10k':
        seq_dataset = GOT10k(config.root_dir_for_GOT_10k, subset='val')
        pair_dataset = Pairwise(seq_dataset)
    elif name == 'VID':
        seq_dataset = ImageNetVID(config.root_dir_for_VID, subset=('train', 'val'))
        pair_dataset = Pairwise(seq_dataset)
    elif name == 'All':
        seq_got_dataset = GOT10k(config.root_dir_for_GOT_10k, subset='train')
        seq_vid_dataset = ImageNetVID(config.root_dir_for_VID, subset=('train', 'val'))
        pair_dataset = Pairwise(seq_got_dataset) + Pairwise(seq_vid_dataset)

    print(len(pair_dataset))

    # setup data loader
    cuda = torch.cuda.is_available()
    loader = DataLoader(pair_dataset,
                        batch_size = config.batch_size,
                        shuffle    = True,
                        pin_memory = cuda,
                        drop_last  = True,
                        num_workers= config.num_workers)

    # setup tracker
    tracker = TrackerSiamFC()

    # training loop
    for epoch in range(config.epoch_num):
        for step, batch in enumerate(loader):

            loss = tracker.step(batch,
                                backward=True,
                                update_lr=(step == 0))

            if step % config.show_step == 0:
                print('Epoch [{}][{}/{}]: Loss: {:.3f}'.format( epoch + 1,
                                                                step  + 1,
                                                                len(loader),
                                                                loss))
                sys.stdout.flush()

        # save checkpoint
        net_path = os.path.join('model', 'model_e%d.pth' % (epoch + 1))
        torch.save(tracker.net.state_dict(), net_path)

        # test on OTB2015 dataset
        tracker_test = TrackerSiamFC(net_path=net_path)
        experiments = ExperimentOTB(config.root_dir_for_OTB, version=2015,
                                    result_dir='{}_dataset/results_{}'.format(name, epoch + 1),
                                    report_dir='{}_dataset/reports_{}'.format(name, epoch + 1))

        # run tracking experiments and report performance
        experiments.run(tracker_test, visualize=False)
        experiments.report([tracker_test.name])
