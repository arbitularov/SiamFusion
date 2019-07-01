from __future__ import absolute_import

from got10k.experiments import *

from siamfc import TrackerSiamFC
from config import config


if __name__ == '__main__':

    # setup tracker
    net_path = 'model/model_e31.pth'

    tracker_test = TrackerSiamFC(net_path=net_path)
    '''experiments = ExperimentOTB(config.root_dir_for_OTB, version=2015,
                                result_dir='dataset/results',
                                report_dir='dataset/reports')'''

    experiments = ExperimentGOT10k('/Users/arbi/Desktop/All2', subset='val',
                                        result_dir='GOT/results',
                                        report_dir='GOT/reports')

    # run tracking experiments and report performance
    experiments.run(tracker_test, visualize=True)
    experiments.report([tracker_test.name])
