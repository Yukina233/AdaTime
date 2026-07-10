import sys
sys.path.append('../../ADATIME/')
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, AUROC, F1Score
import os
import wandb
import pandas as pd
import numpy as np
import warnings
import sklearn.exceptions
import collections

from torchmetrics import Accuracy, AUROC, F1Score
from dataloader.dataloader import data_generator, few_shot_data_generator
from configs.data_model_configs import get_dataset_class
from configs.hparams import get_hparams_class
from configs.sweep_params import sweep_alg_hparams
from utils import fix_randomness, starting_logs, DictAsObject,AverageMeter
from algorithms.algorithms import get_algorithm_class
from models.models import get_backbone_class

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

def count_parameters(model, verbose=True):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    if verbose:
        print(f"{'='*50}")
        print(f"Total params     : {total:>15,}  ({total/1e6:.2f} M)")
        print(f"Trainable params : {trainable:>15,}  ({trainable/1e6:.2f} M)")
        print(f"Frozen params    : {frozen:>15,}  ({frozen/1e6:.2f} M)")
        print(f"{'='*50}")
    return total, trainable

class AbstractTrainer(object):
    """
   This class contain the main training functions for our AdAtime
    """

    def __init__(self, args):
        self.da_method = args.da_method  # Selected  DA Method
        self.dataset = args.dataset  # Selected  Dataset
        self.backbone = args.backbone
        self.device = torch.device(args.device)  # device

        # Exp Description
        self.experiment_description = args.dataset 
        self.run_description = f"{args.da_method}_{args.exp_name}"
        
        # paths
        self.home_path =  os.getcwd() #os.path.dirname(os.getcwd())
        self.save_dir = args.save_dir
        self.data_path = os.path.join(args.data_path, self.dataset)
        # self.create_save_dir(os.path.join(self.home_path,  self.save_dir ))
        self.exp_log_dir = os.path.join(self.home_path, self.save_dir, self.experiment_description, f"{self.run_description}")
        os.makedirs(self.exp_log_dir, exist_ok=True)




        # Specify runs
        self.num_runs = args.num_runs

        # Latent-space visualization settings
        visualize_mds = getattr(args, "visualize_mds", False)
        if isinstance(visualize_mds, str):
            visualize_mds = visualize_mds.lower() in ("1", "true", "yes", "y")
        self.visualize_mds = visualize_mds
        self.mds_max_samples = getattr(args, "mds_max_samples", 1000)
        self.mds_split = getattr(args, "mds_split", "test")
        self.mds_model = getattr(args, "mds_model", "last")
        self.mds_random_state = getattr(args, "mds_random_state", 0)
        self._loaded_checkpoint = None

        # get dataset and base model configs
        self.dataset_configs, self.hparams_class = self.get_configs()

        # to fix dimension of features in classifier and discriminator networks.
        self.dataset_configs.final_out_channels = self.dataset_configs.tcn_final_out_channles if args.backbone == "TCN" else self.dataset_configs.final_out_channels

        # Specify number of hparams
        self.hparams = {**self.hparams_class.alg_hparams[self.da_method],
                                **self.hparams_class.train_params}

        # metrics
        self.num_classes = self.dataset_configs.num_classes
        self.ACC = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.F1 = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")
        self.AUROC = AUROC(task="multiclass", num_classes=self.num_classes)        

        # metrics

    def sweep(self):
        # sweep configurations
        pass
    
    def initialize_algorithm(self):
        # get algorithm class
        algorithm_class = get_algorithm_class(self.da_method)
        backbone_fe = get_backbone_class(self.backbone)

        # Initilaize the algorithm
        self.algorithm = algorithm_class(backbone_fe, self.dataset_configs, self.hparams, self.device)
        count_parameters(self.algorithm)
        self.algorithm.to(self.device)

    def load_checkpoint(self, model_dir):
        checkpoint = torch.load(os.path.join(self.home_path, model_dir, 'checkpoint.pt'), map_location=self.device)
        self._loaded_checkpoint = checkpoint
        last_model = checkpoint['last']
        best_model = checkpoint['best']
        return last_model, best_model

    def load_model_state(self, model_state, checkpoint_name=None):
        visual_state = self._get_visualization_checkpoint_state(checkpoint_name)
        if visual_state is not None and hasattr(self.algorithm, "load_visualization_state"):
            self.algorithm.load_visualization_state(visual_state)
        else:
            self.algorithm.network.load_state_dict(model_state)

    def _get_visualization_checkpoint_state(self, checkpoint_name):
        if checkpoint_name is None or not isinstance(self._loaded_checkpoint, dict):
            return None

        visualization_state = self._loaded_checkpoint.get("visualization")
        if not isinstance(visualization_state, dict):
            return None

        return visualization_state.get(checkpoint_name)

    def train_model(self):
        # Get the algorithm and the backbone network
        algorithm_class = get_algorithm_class(self.da_method)
        backbone_fe = get_backbone_class(self.backbone)

        # Initilaize the algorithm
        self.algorithm = algorithm_class(backbone_fe, self.dataset_configs, self.hparams, self.device)
        self.algorithm.to(self.device)

        # Training the model
        self.last_model, self.best_model = self.algorithm.update(self.src_train_dl, self.trg_train_dl, self.loss_avg_meters, self.logger)
        return self.last_model, self.best_model
    
    def evaluate(self, test_loader):
        feature_extractor = self.algorithm.feature_extractor.to(self.device)
        classifier = self.algorithm.classifier.to(self.device)

        feature_extractor.eval()
        classifier.eval()

        total_loss, preds_list, labels_list = [], [], []

        with torch.no_grad():
            for data, labels in test_loader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                features = feature_extractor(data)
                predictions = classifier(features)

                # compute loss
                loss = F.cross_entropy(predictions, labels)
                total_loss.append(loss.item())
                pred = predictions.detach()  # .argmax(dim=1)  # get the index of the max log-probability

                # append predictions and labels
                preds_list.append(pred)
                labels_list.append(labels)

        self.loss = torch.tensor(total_loss).mean()  # average loss
        self.full_preds = torch.cat((preds_list))
        self.full_labels = torch.cat((labels_list))

    def get_configs(self):
        dataset_class = get_dataset_class(self.dataset)
        hparams_class = get_hparams_class(self.dataset)
        return dataset_class(), hparams_class()

    def load_data(self, src_id, trg_id):
        self.src_train_dl = data_generator(self.data_path, src_id, self.dataset_configs, self.hparams, "train")
        self.src_test_dl = data_generator(self.data_path, src_id, self.dataset_configs, self.hparams, "test")

        self.trg_train_dl = data_generator(self.data_path, trg_id, self.dataset_configs, self.hparams, "train")
        self.trg_test_dl = data_generator(self.data_path, trg_id, self.dataset_configs, self.hparams, "test")

        self.few_shot_dl_5 = few_shot_data_generator(self.trg_test_dl, self.dataset_configs,
                                                     5)  # set 5 to other value if you want other k-shot FST

    def create_save_dir(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    def calculate_metrics_risks(self):
        # calculation based source test data
        self.evaluate(self.src_test_dl)
        src_risk = self.loss.item()
        # calculation based few_shot test data
        self.evaluate(self.few_shot_dl_5)
        fst_risk = self.loss.item()
        # calculation based target test data
        self.evaluate(self.trg_test_dl)
        trg_risk = self.loss.item()

        # calculate metrics
        acc = self.ACC(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        # f1_torch
        f1 = self.F1(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        auroc = self.AUROC(self.full_preds.cpu(), self.full_labels.cpu()).item()
        # f1_sk learn
        # f1 = f1_score(self.full_preds.argmax(dim=1).cpu().numpy(), self.full_labels.cpu().numpy(), average='macro')

        risks = src_risk, fst_risk, trg_risk
        metrics = acc, f1, auroc

        return risks, metrics

    def save_tables_to_file(self,table_results, name):
        # save to file if needed
        table_results.to_csv(os.path.join(self.exp_log_dir,f"{name}.csv"))

    def save_checkpoint(self, home_path, log_dir, last_model, best_model):
        save_dict = {
            "last": last_model,
            "best": best_model
        }
        if hasattr(self, "algorithm") and hasattr(self.algorithm, "get_visualization_checkpoint"):
            save_dict["visualization"] = self.algorithm.get_visualization_checkpoint()
        # save classification report
        save_path = os.path.join(home_path, log_dir, f"checkpoint.pt")
        torch.save(save_dict, save_path)

    def calculate_avg_std_wandb_table(self, results):

        avg_metrics = [np.mean(results.get_column(metric)) for metric in results.columns[2:]]
        std_metrics = [np.std(results.get_column(metric)) for metric in results.columns[2:]]
        summary_metrics = {metric: np.mean(results.get_column(metric)) for metric in results.columns[2:]}

        results.add_data('mean', '-', *avg_metrics)
        results.add_data('std', '-', *std_metrics)

        return results, summary_metrics

    def log_summary_metrics_wandb(self, results, risks):
       
        # Calculate average and standard deviation for metrics
        avg_metrics = [np.mean(results.get_column(metric)) for metric in results.columns[2:]]
        std_metrics = [np.std(results.get_column(metric)) for metric in results.columns[2:]]

        avg_risks = [np.mean(risks.get_column(risk)) for risk in risks.columns[2:]]
        std_risks = [np.std(risks.get_column(risk)) for risk in risks.columns[2:]]

        # Estimate summary metrics
        summary_metrics = {metric: np.mean(results.get_column(metric)) for metric in results.columns[2:]}
        summary_risks = {risk: np.mean(risks.get_column(risk)) for risk in risks.columns[2:]}


        # append avg and std values to metrics
        results.add_data('mean', '-', *avg_metrics)
        results.add_data('std', '-', *std_metrics)

        # append avg and std values to risks 
        results.add_data('mean', '-', *avg_risks)
        risks.add_data('std', '-', *std_risks)

    def wandb_logging(self, total_results, total_risks, summary_metrics, summary_risks):
        # log wandb
        wandb.log({'results': total_results})
        wandb.log({'risks': total_risks})
        wandb.log({'hparams': wandb.Table(dataframe=pd.DataFrame(dict(self.hparams).items(), columns=['parameter', 'value']), allow_mixed_types=True)})
        wandb.log(summary_metrics)
        wandb.log(summary_risks)

    def calculate_metrics(self):
       
        self.evaluate(self.trg_test_dl)
        # accuracy  
        acc = self.ACC(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        # f1
        f1 = self.F1(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        # auroc 
        auroc = self.AUROC(self.full_preds.cpu(), self.full_labels.cpu()).item()

        return acc, f1, auroc

    def calculate_risks(self):
         # calculation based source test data
        self.evaluate(self.src_test_dl)
        src_risk = self.loss.item()
        # calculation based few_shot test data
        self.evaluate(self.few_shot_dl_5)
        fst_risk = self.loss.item()
        # calculation based target test data
        self.evaluate(self.trg_test_dl)
        trg_risk = self.loss.item()

        return src_risk, fst_risk, trg_risk

    def append_results_to_tables(self, table, scenario, run_id, metrics):

        # Create metrics and risks rows
        results_row = [scenario, run_id, *metrics]

        # Create new dataframes for each row
        results_df = pd.DataFrame([results_row], columns=table.columns)

        # Concatenate new dataframes with original dataframes
        table = pd.concat([table, results_df], ignore_index=True)

        return table
    
    def add_mean_std_table(self, table, columns):
        # Calculate average and standard deviation for metrics
        avg_metrics = [table[metric].mean() for metric in columns[2:]]
        std_metrics = [table[metric].std() for metric in columns[2:]]

        # Create dataframes for mean and std values
        mean_metrics_df = pd.DataFrame([['mean', '-', *avg_metrics]], columns=columns)
        std_metrics_df = pd.DataFrame([['std', '-', *std_metrics]], columns=columns)

        # Concatenate original dataframes with mean and std dataframes
        table = pd.concat([table, mean_metrics_df, std_metrics_df], ignore_index=True)

        # Create a formatting function to format each element in the tables
        format_func = lambda x: f"{x:.4f}" if isinstance(x, float) else x

        # Apply the formatting function to each element in the tables
        table = table.applymap(format_func)

        return table

    def calculate_metrics_for_loader(self, loader):
        self.evaluate(loader)
        # accuracy
        acc = self.ACC(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        # f1
        f1 = self.F1(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        # auroc
        auroc = self.AUROC(self.full_preds.cpu(), self.full_labels.cpu()).item()
        return acc, f1, auroc

    def should_visualize_mds_checkpoint(self, checkpoint_name):
        if not self.visualize_mds:
            return False
        return self.mds_model == "both" or self.mds_model == checkpoint_name

    def save_mds_visualization(self, scenario, run_id, checkpoint_name):
        if not self.should_visualize_mds_checkpoint(checkpoint_name):
            return

        try:
            src_loader, trg_loader = self._get_mds_loaders()
            src_encoder, trg_encoder = self._get_mds_encoders()

            src_features, src_labels, src_preds = self._extract_mds_features(src_loader, src_encoder)
            trg_features, trg_labels, trg_preds = self._extract_mds_features(trg_loader, trg_encoder)

            max_per_domain = max(1, int(self.hparams.get("vis_max_samples_per_domain", self.mds_max_samples)))
            src_idx = self._balanced_sample_indices(src_labels, max_per_domain)
            trg_idx = self._balanced_sample_indices(trg_labels, max_per_domain)

            src_features = src_features[src_idx]
            src_labels = src_labels[src_idx]
            src_preds = src_preds[src_idx]
            trg_features = trg_features[trg_idx]
            trg_labels = trg_labels[trg_idx]
            trg_preds = trg_preds[trg_idx]

            features = np.concatenate([src_features, trg_features], axis=0)
            labels = np.concatenate([src_labels, trg_labels], axis=0).astype(int)
            preds = np.concatenate([src_preds, trg_preds], axis=0).astype(int)
            domains = np.asarray(["Source"] * len(src_features) + ["Target"] * len(trg_features))

            if len(features) < 5:
                self._log_visualization_message(f"[vis] skipped: too few samples ({len(features)})")
                return

            coords = self._compute_mds_coordinates(features)
            output_dir = os.path.join(self._scenario_output_dir(), "visualizations")
            os.makedirs(output_dir, exist_ok=True)

            src_id, trg_id = self._split_scenario_name(scenario)
            checkpoint_suffix = f"_{checkpoint_name}" if self.mds_model == "both" else ""
            base_name = f"{self.da_method}_{src_id}_to_{trg_id}_run_{run_id}{checkpoint_suffix}_latent_mds"
            csv_path = os.path.join(output_dir, f"{base_name}.csv")
            fig_path = os.path.join(output_dir, f"{base_name}.png")

            self._plot_mds_coordinates(coords, domains, labels, scenario, run_id, checkpoint_name, fig_path)
            pd.DataFrame({
                "mds_1": coords[:, 0],
                "mds_2": coords[:, 1],
                "domain": domains,
                "label": labels,
                "pred": preds,
            }).to_csv(csv_path, index=False)

            self._log_visualization_message(f"[vis] saved latent MDS to {fig_path}")
        except Exception as exc:
            self._log_visualization_message(f"[vis] failed {scenario} run {run_id} {checkpoint_name}: {exc}")

    def _get_mds_loaders(self):
        if self.mds_split == "train":
            return self.src_train_dl, self.trg_train_dl
        return self.src_test_dl, self.trg_test_dl

    def _get_mds_encoders(self):
        if hasattr(self.algorithm, "visualization_encoders"):
            return self.algorithm.visualization_encoders()
        return self.algorithm.feature_extractor, self.algorithm.feature_extractor

    def _extract_mds_features(self, loader, encoder):
        encoder = encoder.to(self.device)
        classifier = self.algorithm.classifier.to(self.device)
        encoder.eval()
        classifier.eval()

        features_list, labels_list, preds_list = [], [], []
        with torch.no_grad():
            for data, labels in loader:
                data = data.float().to(self.device)
                features = encoder(data).flatten(1)
                logits = classifier(features)
                preds = logits.argmax(dim=1).detach().cpu()

                if labels is None:
                    labels = torch.full((data.size(0),), -1, dtype=torch.long)
                else:
                    labels = labels.view((-1)).long().cpu()

                features_list.append(features.detach().cpu())
                labels_list.append(labels)
                preds_list.append(preds)

        features = torch.cat(features_list, dim=0).numpy()
        labels = torch.cat(labels_list, dim=0).numpy()
        preds = torch.cat(preds_list, dim=0).numpy()
        return features, labels, preds

    def _balanced_sample_indices(self, labels, max_count):
        if max_count <= 0 or len(labels) <= max_count:
            return np.arange(len(labels))

        rng = np.random.default_rng(self.mds_random_state)
        selected = []
        classes = np.unique(labels)
        per_class = max(1, max_count // max(1, len(classes)))

        for label in classes:
            class_indices = np.flatnonzero(labels == label)
            take = min(per_class, len(class_indices))
            if take > 0:
                selected.extend(rng.choice(class_indices, size=take, replace=False).tolist())

        if len(selected) < max_count:
            selected_arr = np.asarray(selected, dtype=int)
            remaining = np.setdiff1d(np.arange(len(labels)), selected_arr, assume_unique=False)
            fill_count = min(max_count - len(selected), len(remaining))
            if fill_count > 0:
                selected.extend(rng.choice(remaining, size=fill_count, replace=False).tolist())

        selected = np.asarray(selected[:max_count], dtype=int)
        rng.shuffle(selected)
        return selected

    def _compute_mds_coordinates(self, features):
        from sklearn.manifold import MDS
        from sklearn.preprocessing import StandardScaler

        features = StandardScaler().fit_transform(features)
        mds = MDS(n_components=2,
                  random_state=int(self.hparams.get("vis_mds_seed", self.mds_random_state)),
                  max_iter=int(self.hparams.get("vis_mds_max_iter", 300)),
                  n_init=int(self.hparams.get("vis_mds_n_init", 4)),
                  dissimilarity="euclidean")
        return mds.fit_transform(features)

    def _plot_mds_coordinates(self, coords, domains, labels, scenario, run_id, checkpoint_name, fig_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        src_id, trg_id = self._split_scenario_name(scenario)
        cmap = plt.get_cmap("tab10", self.num_classes) if self.num_classes <= 10 else plt.get_cmap("tab20", self.num_classes)
        colors = [cmap(i) for i in range(self.num_classes)]
        markers = {"Source": "o", "Target": "x"}
        sizes = {"Source": 24, "Target": 42}

        fig, ax = plt.subplots(figsize=(6, 6))
        for domain in ("Source", "Target"):
            for cls in range(self.num_classes):
                mask = (domains == domain) & (labels == cls)
                if not np.any(mask):
                    continue
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    c=[colors[cls]],
                    marker=markers[domain],
                    s=sizes[domain],
                    linewidths=1.2 if domain == "Target" else 0.35,
                    edgecolors="black" if domain == "Source" else None,
                    alpha=0.78 if domain == "Source" else 0.9,
                )

        class_handles = []
        for cls in range(self.num_classes):
            class_handles.append(
                Line2D([0], [0], marker="o", linestyle="", markerfacecolor=colors[cls],
                       markeredgecolor="black", markersize=8, label=self._label_to_class_name(cls))
            )
        domain_handles = [
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor="gray",
                   markeredgecolor="black", markersize=8, label="Source"),
            Line2D([0], [0], marker="x", linestyle="", markeredgecolor="gray",
                   markeredgewidth=1.8, markersize=9, label="Target"),
        ]

        leg1 = ax.legend(handles=class_handles, title="Class", loc="upper right", framealpha=0.92)
        ax.add_artist(leg1)
        ax.legend(handles=domain_handles, title="Domain", loc="lower right", framealpha=0.92)
        title_suffix = f", {checkpoint_name}" if self.mds_model == "both" else ""
        ax.set_title(f"{self.da_method}: latent MDS ({src_id} to {trg_id}, run {run_id}{title_suffix})")
        ax.set_xlabel("MDS 1")
        ax.set_ylabel("MDS 2")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_path, dpi=330, bbox_inches="tight")
        plt.close(fig)

    def _label_to_class_name(self, label):
        class_names = getattr(self.dataset_configs, "class_names", None)
        label = int(label)
        if class_names is not None and 0 <= label < len(class_names):
            return class_names[label]
        return f"class {label}"

    def _scenario_output_dir(self):
        if os.path.isabs(self.scenario_log_dir):
            return self.scenario_log_dir
        return os.path.join(self.home_path, self.scenario_log_dir)

    def _split_scenario_name(self, scenario):
        if "_to_" in scenario:
            return tuple(scenario.split("_to_", 1))
        return scenario, "target"

    def _log_visualization_message(self, msg):
        if hasattr(self, "logger") and self.logger is not None:
            self.logger.debug(msg)
        print(msg)

