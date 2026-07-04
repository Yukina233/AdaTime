import sys
import time

import torch
import torch.nn.functional as F
import os
import wandb
import pandas as pd
import numpy as np
import warnings
import sklearn.exceptions
import collections
import argparse
import warnings
import sklearn.exceptions

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from utils import fix_randomness, starting_logs, AverageMeter
from algorithms.algorithms import get_algorithm_class
from models.models import get_backbone_class
from trainers.abstract_trainer import AbstractTrainer
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
parser = argparse.ArgumentParser()
       
def insert_mean_std(df):
    acc_mean = df['acc']['mean'].mean()
    acc_std_mean = df['acc']['std'].mean()
    f1_mean = df['f1_score']['mean'].mean()
    f1_std_mean = df['f1_score']['std'].mean()
    auroc_mean = df['auroc']['mean'].mean()
    auroc_std_mean = df['auroc']['std'].mean()
    df.loc['mean'] = [acc_mean, acc_std_mean, f1_mean, f1_std_mean, auroc_mean, auroc_std_mean]

class Trainer(AbstractTrainer):
    """
   This class contain the main training functions for our AdAtime
    """

    def __init__(self, args):
        super().__init__(args)

        self.results_columns = ["scenario", "run", "acc", "f1_score", "auroc"]
        self.risks_columns = ["scenario", "run", "src_risk", "few_shot_risk", "trg_risk"]


    def fit(self):

        # table with metrics
        table_results = pd.DataFrame(columns=self.results_columns)

        # table with risks
        table_risks = pd.DataFrame(columns=self.risks_columns)


        # Trainer
        for src_id, trg_id in self.dataset_configs.scenarios:
            for run_id in range(self.num_runs):
                # fixing random seed
                fix_randomness(run_id)

                # Logging
                self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.da_method, self.exp_log_dir,
                                                                src_id, trg_id, run_id)
                # Average meters
                self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

                # Load data
                self.load_data(src_id, trg_id)
                
                # initiate the domain adaptation algorithm
                self.initialize_algorithm()

                start_time = time.time()

                # Train the domain adaptation algorithm
                self.last_model, self.best_model = self.algorithm.update(self.src_train_dl, self.trg_train_dl, self.loss_avg_meters, self.logger)

                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"程序运行时间: {elapsed_time:.4f} 秒")

                # Save checkpoint
                self.save_checkpoint(self.home_path, self.scenario_log_dir, self.last_model, self.best_model)

                # Calculate risks and metrics
                metrics = self.calculate_metrics()
                risks = self.calculate_risks()

                # Append results to tables
                scenario = f"{src_id}_to_{trg_id}"
                table_results = self.append_results_to_tables(table_results, scenario, run_id, metrics)
                table_risks = self.append_results_to_tables(table_risks, scenario, run_id, risks)

        # Calculate and append mean and std to tables
        table_results = self.add_mean_std_table(table_results, self.results_columns)
        table_risks = self.add_mean_std_table(table_risks, self.risks_columns)


        # Save tables to file if needed
        self.save_tables_to_file(table_results, 'results')
        self.save_tables_to_file(table_risks, 'risks')

    def test(self):
        # Results dataframes
        last_results = pd.DataFrame(columns=self.results_columns)
        best_results = pd.DataFrame(columns=self.results_columns)

        # Cross-domain scenarios
        for src_id, trg_id in self.dataset_configs.scenarios:
            for run_id in range(self.num_runs):
                # fixing random seed
                fix_randomness(run_id)

                # Logging
                self.scenario_log_dir = os.path.join(self.exp_log_dir, src_id + "_to_" + trg_id + "_run_" + str(run_id))

                self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

                # Load data
                self.load_data(src_id, trg_id)

                # Build model
                self.initialize_algorithm()

                # Load chechpoint 
                last_chk, best_chk = self.load_checkpoint(self.scenario_log_dir)

                # Testing the last model
                self.algorithm.network.load_state_dict(last_chk)
                self.evaluate(self.trg_test_dl)
                last_metrics = self.calculate_metrics()
                last_results = self.append_results_to_tables(last_results, f"{src_id}_to_{trg_id}", run_id,
                                                             last_metrics)
                

                # Testing the best model
                self.algorithm.network.load_state_dict(best_chk)
                self.evaluate(self.trg_test_dl)
                best_metrics = self.calculate_metrics()
                # Append results to tables
                best_results = self.append_results_to_tables(best_results, f"{src_id}_to_{trg_id}", run_id,
                                                             best_metrics)

        last_scenario_mean_std = last_results.groupby('scenario')[['acc', 'f1_score', 'auroc']].agg(['mean', 'std'])
        insert_mean_std(last_scenario_mean_std)

        best_scenario_mean_std = best_results.groupby('scenario')[['acc', 'f1_score', 'auroc']].agg(['mean', 'std'])
        insert_mean_std(best_scenario_mean_std)



        # Save tables to file if needed
        self.save_tables_to_file(last_scenario_mean_std, 'last_results')
        self.save_tables_to_file(best_scenario_mean_std, 'best_results')

        # printing summary 
        summary_last = {metric: np.mean(last_results[metric]) for metric in self.results_columns[2:]}
        summary_best = {metric: np.mean(best_results[metric]) for metric in self.results_columns[2:]}
        for summary_name, summary in [('Last', summary_last), ('Best', summary_best)]:
            for key, val in summary.items():
                print(f'{summary_name}: {key}\t: {val:2.4f}')

    def test_with_source_and_target(self):
        # Results dataframes (target domain)
        last_results = pd.DataFrame(columns=self.results_columns)
        best_results = pd.DataFrame(columns=self.results_columns)

        # Results dataframes (source domain)
        last_src_results = pd.DataFrame(columns=self.results_columns)
        best_src_results = pd.DataFrame(columns=self.results_columns)

        # Cross-domain scenarios
        for src_id, trg_id in self.dataset_configs.scenarios:
            for run_id in range(self.num_runs):
                # fixing random seed
                fix_randomness(run_id)

                # Logging
                self.scenario_log_dir = os.path.join(self.exp_log_dir, src_id + "_to_" + trg_id + "_run_" + str(run_id))

                self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

                # Load data
                self.load_data(src_id, trg_id)

                # Build model
                self.initialize_algorithm()

                # Load checkpoint
                last_chk, best_chk = self.load_checkpoint(self.scenario_log_dir)

                scenario = f"{src_id}_to_{trg_id}"

                # ---------------- Testing the last model ----------------
                self.algorithm.network.load_state_dict(last_chk)
                # target domain
                last_metrics = self.calculate_metrics_for_loader(self.trg_test_dl)
                last_results = self.append_results_to_tables(last_results, scenario, run_id, last_metrics)
                # source domain
                last_src_metrics = self.calculate_metrics_for_loader(self.src_test_dl)
                last_src_results = self.append_results_to_tables(last_src_results, scenario, run_id, last_src_metrics)

                # ---------------- Testing the best model ----------------
                self.algorithm.network.load_state_dict(best_chk)
                # target domain
                best_metrics = self.calculate_metrics_for_loader(self.trg_test_dl)
                best_results = self.append_results_to_tables(best_results, scenario, run_id, best_metrics)
                # source domain
                best_src_metrics = self.calculate_metrics_for_loader(self.src_test_dl)
                best_src_results = self.append_results_to_tables(best_src_results, scenario, run_id, best_src_metrics)

        # ---------------- Aggregate: target domain ----------------
        last_scenario_mean_std = last_results.groupby('scenario')[['acc', 'f1_score', 'auroc']].agg(['mean', 'std'])
        insert_mean_std(last_scenario_mean_std)

        best_scenario_mean_std = best_results.groupby('scenario')[['acc', 'f1_score', 'auroc']].agg(['mean', 'std'])
        insert_mean_std(best_scenario_mean_std)

        # ---------------- Aggregate: source domain ----------------
        last_src_scenario_mean_std = last_src_results.groupby('scenario')[['acc', 'f1_score', 'auroc']].agg(
            ['mean', 'std'])
        insert_mean_std(last_src_scenario_mean_std)

        best_src_scenario_mean_std = best_src_results.groupby('scenario')[['acc', 'f1_score', 'auroc']].agg(
            ['mean', 'std'])
        insert_mean_std(best_src_scenario_mean_std)

        # ---------------- Save tables to file ----------------
        # target domain
        self.save_tables_to_file(last_scenario_mean_std, 'last_results')
        self.save_tables_to_file(best_scenario_mean_std, 'best_results')
        # source domain
        self.save_tables_to_file(last_src_scenario_mean_std, 'last_src_results')
        self.save_tables_to_file(best_src_scenario_mean_std, 'best_src_results')

        # ---------------- Print summary ----------------
        summary_last = {metric: np.mean(last_results[metric]) for metric in self.results_columns[2:]}
        summary_best = {metric: np.mean(best_results[metric]) for metric in self.results_columns[2:]}
        summary_last_src = {metric: np.mean(last_src_results[metric]) for metric in self.results_columns[2:]}
        summary_best_src = {metric: np.mean(best_src_results[metric]) for metric in self.results_columns[2:]}

        print("==================== Target Domain ====================")
        for summary_name, summary in [('Last', summary_last), ('Best', summary_best)]:
            for key, val in summary.items():
                print(f'{summary_name}: {key}\t: {val:2.4f}')

        print("==================== Source Domain ====================")
        for summary_name, summary in [('Last', summary_last_src), ('Best', summary_best_src)]:
            for key, val in summary.items():
                print(f'{summary_name}: {key}\t: {val:2.4f}')


