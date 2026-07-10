from trainers.train import Trainer
from datetime import datetime

import argparse

def get_time():
    return datetime.now().strftime('%d_%m_%Y_%H_%M_%S')

parser = argparse.ArgumentParser()

if __name__ == "__main__":

    # ========  Experiments Phase ================
    parser.add_argument('--phase', default='train', type=str, help='train, test')

    # ========  Experiments Name ================
    parser.add_argument('--save_dir', default='experiments_logs', type=str, help='Directory containing all experiments')
    parser.add_argument('--exp_name', default=f'{get_time()}', type=str, help='experiment name')

    # ========= Select the DA methods ============0.6
    parser.add_argument('--da_method', default='ADDA', type=str,
                        help='NO_ADAPT, Deep_Coral, MMDA, DANN, CDAN, DIRT, DSAN, HoMM, CoDATS, AdvSKM, SASA, CoTMix, ADDA, TARGET_ONLY')

    # ========= Select the DATASET ==============
    parser.add_argument('--data_path', default=r'../ADATIME_data', type=str, help='Path containing datase2t')
    parser.add_argument('--dataset', default='HHAR', type=str, help='Dataset of choice: (WISDM - EEG - HAR - HHAR_SA)')

    # ========= Select the BACKBONE ==============
    parser.add_argument('--backbone', default='CNN', type=str, help='Backbone of choice: (CNN - RESNET18 - TCN)')

    # ========= Experiment settings ===============
    parser.add_argument('--num_runs', default=1, type=int, help='Number of consecutive run with different seeds')
    parser.add_argument('--device', default="cuda", type=str, help='cpu or cuda')

    # ========= Latent-space visualization =========
    parser.add_argument('--visualize_mds', default='true',
                        help='Save MDS latent-space plots with domain markers and class colors')
    parser.add_argument('--mds_max_samples', default=1000, type=int,
                        help='Maximum samples per domain used in each MDS plot')
    parser.add_argument('--mds_split', default='test', type=str, choices=['train', 'test'],
                        help='Data split used for MDS visualization')
    parser.add_argument('--mds_model', default='last', type=str, choices=['last', 'best', 'both'],
                        help='Checkpoint state visualized by MDS')
    parser.add_argument('--mds_random_state', default=0, type=int,
                        help='Random seed for balanced MDS sampling and projection')

    # arguments
    args = parser.parse_args()

    # create trainier object
    trainer = Trainer(args)

    # train and test
    if args.phase == 'train':
        trainer.fit()
        trainer.test_with_source_and_target()
    elif args.phase == 'test':
        trainer.test_with_source_and_target()
# TODO:
# 1- Change the naming of the functions ---> ( Done)
# 2- Change the algorithms following DCORAL --> (Done)
# 3- Keep one trainer for both train and test -->(Done)
# 4- Create the new joint loader that consider the all possible batches --> Done
# 5- Implement Lower/Upper Bound Approach --> Done
# 6- Add the best hparams --> Done
# 7- Add pretrain based methods (ADDA, MCD, MDD)
