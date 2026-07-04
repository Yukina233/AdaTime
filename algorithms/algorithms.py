import torch
import torch.nn as nn
import numpy as np
import itertools

from models.models import classifier, ReverseLayerF, Discriminator, RandomLayer, Discriminator_CDAN, \
    codats_classifier, AdvSKM_Disc, CNN_ATTN, CNN_Decoder
from models.loss import MMD_loss, CORAL, ConditionalEntropyLoss, VAT, LMMD_loss, HoMM_loss, NTXentLoss, SupConLoss
from utils import EMA
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
import torch.nn. functional as F

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain adaptation algorithm.
    Subclasses should implement the update() method.
    """

    def __init__(self, configs, backbone):
        super(Algorithm, self).__init__()
        self.configs = configs

        self.cross_entropy = nn.CrossEntropyLoss()
        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)


    # update function is common to all algorithms
    def update(self, src_loader, trg_loader, avg_meter, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            if hasattr(self, "current_epoch"):
                self.current_epoch = epoch - 1
            # training loop 
            self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())


            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        last_model = self.network.state_dict()

        return last_model, best_model

    # train loop vary from one method to another
    def training_epoch(self, *args, **kwargs):
        raise NotImplementedError

def contrastive_loss(features, labels, margin=1.0):
    """
    Contrastive loss for feature regularization.
    Args:
        features: Tensor, shape [N, D], feature representations.
        labels: Tensor, shape [N], ground truth labels.
        margin: float, margin for contrastive loss.
    Returns:
        loss: Scalar tensor, the contrastive loss.
    """
    pairwise_dist = torch.cdist(features, features, p=2)  # Pairwise Euclidean distance, shape [N, N]
    labels = labels.unsqueeze(1)  # Shape [N, 1]
    pairwise_labels = (labels == labels.T).float()  # Shape [N, N], 1 for positive pairs, 0 for negative pairs

    # Positive pairs loss
    positive_loss = pairwise_labels * pairwise_dist**2

    # Negative pairs loss
    negative_loss = (1 - pairwise_labels) * F.relu(margin - pairwise_dist)**2

    # Combine losses
    loss = (positive_loss + negative_loss).mean()
    return loss

class NO_ADAPT(Algorithm):
    """
    Lower bound: train on source and test on target.
    """
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):
        for src_x, src_y in src_loader:

            src_x, src_y = src_x.to(self.device), src_y.to(self.device)
            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            src_cls_loss = self.cross_entropy(src_pred, src_y)

            loss = src_cls_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses = {'Src_cls_loss': src_cls_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()


class TARGET_ONLY(Algorithm):
    """
    Upper bound: train on target and test on target.
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):

        for trg_x, trg_y in trg_loader:

            trg_x, trg_y = trg_x.to(self.device), trg_y.to(self.device)

            trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.classifier(trg_feat)

            trg_cls_loss = self.cross_entropy(trg_pred, trg_y)

            loss = trg_cls_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses = {'Trg_cls_loss': trg_cls_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()


class Deep_Coral(Algorithm):
    """
    Deep Coral: https://arxiv.org/abs/1607.01719
    """
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # correlation alignment loss
        self.coral = CORAL()


    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        # add if statement

        if len(src_loader) > len(trg_loader):
            joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        else:
            joint_loader =enumerate(zip(itertools.cycle(src_loader), trg_loader))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            src_cls_loss = self.cross_entropy(src_pred, src_y)

            trg_feat = self.feature_extractor(trg_x)

            coral_loss = self.coral(src_feat, trg_feat)

            loss = self.hparams["coral_wt"] * coral_loss + \
                self.hparams["src_cls_loss_wt"] * src_cls_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses = {'Total_loss': loss.item(), 'Src_cls_loss': src_cls_loss.item(),
                    'coral_loss': coral_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()

class MMDA(Algorithm):
    """
    MMDA: https://arxiv.org/abs/1901.00282
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Aligment losses
        self.mmd = MMD_loss()
        self.coral = CORAL()
        self.cond_ent = ConditionalEntropyLoss()


    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            src_cls_loss = self.cross_entropy(src_pred, src_y)

            trg_feat = self.feature_extractor(trg_x)
            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            src_cls_loss = self.cross_entropy(src_pred, src_y)

            trg_feat = self.feature_extractor(trg_x)

            coral_loss = self.coral(src_feat, trg_feat)
            mmd_loss = self.mmd(src_feat, trg_feat)
            cond_ent_loss = self.cond_ent(trg_feat)

            loss = self.hparams["coral_wt"] * coral_loss + \
                self.hparams["mmd_wt"] * mmd_loss + \
                self.hparams["cond_ent_wt"] * cond_ent_loss + \
                self.hparams["src_cls_loss_wt"] * src_cls_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(), 'Coral_loss': coral_loss.item(), 'MMD_loss': mmd_loss.item(),
                    'cond_ent_wt': cond_ent_loss.item(), 'Src_cls_loss': src_cls_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()


class DANN(Algorithm):
    """
    DANN: https://arxiv.org/abs/1505.07818
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)


        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Domain Discriminator
        self.domain_classifier = Discriminator(configs)
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):
        # Combine dataloaders
        # Method 1 (min len of both domains)
        # joint_loader = enumerate(zip(src_loader, trg_loader))

        # Method 2 (max len of both domains)
        # joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        num_batches = max(len(src_loader), len(trg_loader))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:

            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)

            p = float(step + epoch * num_batches) / self.hparams["num_epochs"] + 1 / num_batches
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # zero grad
            self.optimizer.zero_grad()
            self.optimizer_disc.zero_grad()

            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            trg_feat = self.feature_extractor(trg_x)

            # Task classification  Loss
            src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

            # Domain classification loss
            # source
            src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
            src_domain_pred = self.domain_classifier(src_feat_reversed)
            src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())

            # target
            trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
            trg_domain_pred = self.domain_classifier(trg_feat_reversed)
            trg_domain_loss = self.cross_entropy(trg_domain_pred, domain_label_trg.long())

            # Total domain loss
            domain_loss = src_domain_loss + trg_domain_loss

            loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + \
                self.hparams["domain_loss_wt"] * domain_loss

            loss.backward()
            self.optimizer.step()
            self.optimizer_disc.step()

            losses =  {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()

class DANN_DC(Algorithm):
    """
    DANN_DC: DANN with distance constraints https://arxiv.org/abs/1505.07818
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)


        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Domain Discriminator
        self.domain_classifier = Discriminator(configs)
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):
        # Combine dataloaders
        # Method 1 (min len of both domains)
        # joint_loader = enumerate(zip(src_loader, trg_loader))

        # Method 2 (max len of both domains)
        # joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        num_batches = max(len(src_loader), len(trg_loader))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:

            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)

            p = float(step + epoch * num_batches) / self.hparams["num_epochs"] + 1 / num_batches
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # zero grad
            self.optimizer.zero_grad()
            self.optimizer_disc.zero_grad()

            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            trg_feat = self.feature_extractor(trg_x)

            # Task classification  Loss
            src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)


            # Domain classification loss
            # source
            src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
            src_domain_pred = self.domain_classifier(src_feat_reversed)
            src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())

            # target
            trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
            trg_domain_pred = self.domain_classifier(trg_feat_reversed)
            trg_domain_loss = self.cross_entropy(trg_domain_pred, domain_label_trg.long())

            # Total domain loss
            domain_loss = src_domain_loss + trg_domain_loss

            loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + \
                self.hparams["domain_loss_wt"] * domain_loss

            loss.backward()
            self.optimizer.step()
            self.optimizer_disc.step()

            losses =  {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()

class DANN_CL(Algorithm):
    """
    DANN_CL: DANN with Contrastive Learning https://arxiv.org/abs/1505.07818
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)


        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Domain Discriminator
        self.domain_classifier = Discriminator(configs)
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):
        # Combine dataloaders
        # Method 1 (min len of both domains)
        # joint_loader = enumerate(zip(src_loader, trg_loader))

        # Method 2 (max len of both domains)
        # joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        num_batches = max(len(src_loader), len(trg_loader))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:

            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)

            p = float(step + epoch * num_batches) / self.hparams["num_epochs"] + 1 / num_batches
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # zero grad
            self.optimizer.zero_grad()
            self.optimizer_disc.zero_grad()

            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            trg_feat = self.feature_extractor(trg_x)

            # Task classification loss
            src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

            # Contrastive loss for source features
            contrastive_loss_src = contrastive_loss(src_feat, src_y, margin=self.hparams["contrastive_margin"])

            # Combine classification loss and contrastive loss
            task_loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + \
                        self.hparams["contrastive_loss_wt"] * contrastive_loss_src

            # Domain classification loss
            # source
            src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
            src_domain_pred = self.domain_classifier(src_feat_reversed)
            src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())

            # target
            trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
            trg_domain_pred = self.domain_classifier(trg_feat_reversed)
            trg_domain_loss = self.cross_entropy(trg_domain_pred, domain_label_trg.long())

            # Total domain loss
            domain_loss = src_domain_loss + trg_domain_loss

            # Total loss
            loss = task_loss + self.hparams["domain_loss_wt"] * domain_loss

            loss.backward()
            self.optimizer.step()
            self.optimizer_disc.step()

            losses = {
                'Total_loss': loss.item(),
                'Domain_loss': domain_loss.item(),
                'Src_cls_loss': src_cls_loss.item(),
                'Contrastive_loss': contrastive_loss_src.item()
            }

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()

class CDAN(Algorithm):
    """
    CDAN: https://arxiv.org/abs/1705.10667
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)


        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Aligment Losses
        self.criterion_cond = ConditionalEntropyLoss().to(device)

        self.domain_classifier = Discriminator_CDAN(configs)
        self.random_layer = RandomLayer([configs.features_len * configs.final_out_channels, configs.num_classes],
                                        configs.features_len * configs.final_out_channels)
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"])

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)
            # prepare true domain labels
            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)
            domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0).long()

            # source features and predictions
            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            # target features and predictions
            trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.classifier(trg_feat)

            # concatenate features and predictions
            feat_concat = torch.cat((src_feat, trg_feat), dim=0)
            pred_concat = torch.cat((src_pred, trg_pred), dim=0)

            # Domain classification loss
            feat_x_pred = torch.bmm(pred_concat.unsqueeze(2), feat_concat.unsqueeze(1)).detach()
            disc_prediction = self.domain_classifier(feat_x_pred.view(-1, pred_concat.size(1) * feat_concat.size(1)))
            disc_loss = self.cross_entropy(disc_prediction, domain_label_concat)

            # update Domain classification
            self.optimizer_disc.zero_grad()
            disc_loss.backward()
            self.optimizer_disc.step()

            # prepare fake domain labels for training the feature extractor
            domain_label_src = torch.zeros(len(src_x)).long().to(self.device)
            domain_label_trg = torch.ones(len(trg_x)).long().to(self.device)
            domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0)

            # Repeat predictions after updating discriminator
            feat_x_pred = torch.bmm(pred_concat.unsqueeze(2), feat_concat.unsqueeze(1))
            disc_prediction = self.domain_classifier(feat_x_pred.view(-1, pred_concat.size(1) * feat_concat.size(1)))
            # loss of domain discriminator according to fake labels

            domain_loss = self.cross_entropy(disc_prediction, domain_label_concat)

            # Task classification  Loss
            src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

            # conditional entropy loss.
            loss_trg_cent = self.criterion_cond(trg_pred)

            # total loss
            loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + self.hparams["domain_loss_wt"] * domain_loss + \
                self.hparams["cond_ent_wt"] * loss_trg_cent

            # update feature extractor
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item(),
                    'cond_ent_loss': loss_trg_cent.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)
        self.lr_scheduler.step()

class DIRT(Algorithm):
    """
    DIRT-T: https://arxiv.org/abs/1802.08735
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device


        # Aligment losses
        self.criterion_cond = ConditionalEntropyLoss().to(device)
        self.vat_loss = VAT(self.network, device).to(device)
        self.ema = EMA(0.998)
        self.ema.register(self.network)

        # Discriminator
        self.domain_classifier = Discriminator(configs)
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)
            # prepare true domain labels
            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)
            domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0).long()

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            # target features and predictions
            trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.classifier(trg_feat)

            # concatenate features and predictions
            feat_concat = torch.cat((src_feat, trg_feat), dim=0)

            # Domain classification loss
            disc_prediction = self.domain_classifier(feat_concat.detach())
            disc_loss = self.cross_entropy(disc_prediction, domain_label_concat)

            # update Domain classification
            self.optimizer_disc.zero_grad()
            disc_loss.backward()
            self.optimizer_disc.step()

            # prepare fake domain labels for training the feature extractor
            domain_label_src = torch.zeros(len(src_x)).long().to(self.device)
            domain_label_trg = torch.ones(len(trg_x)).long().to(self.device)
            domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0)

            # Repeat predictions after updating discriminator
            disc_prediction = self.domain_classifier(feat_concat)

            # loss of domain discriminator according to fake labels
            domain_loss = self.cross_entropy(disc_prediction, domain_label_concat)

            # Task classification  Loss
            src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

            # conditional entropy loss.
            loss_trg_cent = self.criterion_cond(trg_pred)

            # Virual advariarial training loss
            loss_src_vat = self.vat_loss(src_x, src_pred)
            loss_trg_vat = self.vat_loss(trg_x, trg_pred)
            total_vat = loss_src_vat + loss_trg_vat
            # total loss
            loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + self.hparams["domain_loss_wt"] * domain_loss + \
                self.hparams["cond_ent_wt"] * loss_trg_cent + self.hparams["vat_loss_wt"] * total_vat

            # update exponential moving average
            self.ema(self.network)

            # update feature extractor
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item(),
                    'cond_ent_loss': loss_trg_cent.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()

class DSAN(Algorithm):
    """
    DSAN: https://ieeexplore.ieee.org/document/9085896
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Alignment losses
        self.loss_LMMD = LMMD_loss(device=device, class_num=configs.num_classes).to(device)

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)        # extract source features
            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            # extract target features
            trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.classifier(trg_feat)

            # calculate lmmd loss
            domain_loss = self.loss_LMMD.get_loss(src_feat, trg_feat, src_y, torch.nn.functional.softmax(trg_pred, dim=1))

            # calculate source classification loss
            src_cls_loss = self.cross_entropy(src_pred, src_y)

            # calculate the total loss
            loss = self.hparams["domain_loss_wt"] * domain_loss + \
                self.hparams["src_cls_loss_wt"] * src_cls_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(), 'LMMD_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()

class HoMM(Algorithm):
    """
    HoMM: https://arxiv.org/pdf/1912.11976.pdf
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # aligment losses
        self.coral = CORAL()
        self.HoMM_loss = HoMM_loss()

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)           # extract source features

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            # extract target features
            trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.classifier(trg_feat)

            # calculate source classification loss
            src_cls_loss = self.cross_entropy(src_pred, src_y)

            # calculate lmmd loss
            domain_loss = self.HoMM_loss(src_feat, trg_feat)

            # calculate the total loss
            loss = self.hparams["domain_loss_wt"] * domain_loss + \
                self.hparams["src_cls_loss_wt"] * src_cls_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(), 'HoMM_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()


class DDC(Algorithm):
    """
    DDC: https://arxiv.org/abs/1412.3474
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Aligment losses
        self.mmd_loss = MMD_loss()

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)           # extract source features
            # extract source features
            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            # extract target features
            trg_feat = self.feature_extractor(trg_x)

            # calculate source classification loss
            src_cls_loss = self.cross_entropy(src_pred, src_y)

            # calculate mmd loss
            domain_loss = self.mmd_loss(src_feat, trg_feat)

            # calculate the total loss
            loss = self.hparams["domain_loss_wt"] * domain_loss + \
                self.hparams["src_cls_loss_wt"] * src_cls_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(), 'MMD_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()

class CoDATS(Algorithm):
    """
    CoDATS: https://arxiv.org/pdf/2005.10996.pdf
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # we replace the original classifier with codats the classifier
        # remember to use same name of self.classifier, as we use it for the model evaluation
        self.classifier = codats_classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device


        # Domain classifier
        self.domain_classifier = Discriminator(configs)

        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        num_batches = max(len(src_loader), len(trg_loader))
        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)           # extract source features

            p = float(step + epoch * num_batches) / self.hparams["num_epochs"] + 1 / num_batches
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # zero grad
            self.optimizer.zero_grad()
            self.optimizer_disc.zero_grad()

            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            trg_feat = self.feature_extractor(trg_x)

            # Task classification  Loss
            src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

            # Domain classification loss
            # source
            src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
            src_domain_pred = self.domain_classifier(src_feat_reversed)
            src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())

            # target
            trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
            trg_domain_pred = self.domain_classifier(trg_feat_reversed)
            trg_domain_loss = self.cross_entropy(trg_domain_pred, domain_label_trg.long())

            # Total domain loss
            domain_loss = src_domain_loss + trg_domain_loss

            loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + \
                self.hparams["domain_loss_wt"] * domain_loss

            loss.backward()
            self.optimizer.step()
            self.optimizer_disc.step()

            losses =  {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}
            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()

class AdvSKM(Algorithm):
    """
    AdvSKM: https://www.ijcai.org/proceedings/2021/0378.pdf
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Aligment losses
        self.mmd_loss = MMD_loss()
        self.AdvSKM_embedder = AdvSKM_Disc(configs).to(device)
        self.optimizer_disc = torch.optim.Adam(
            self.AdvSKM_embedder.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)         # extract source features

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            # extract target features
            trg_feat = self.feature_extractor(trg_x)

            source_embedding_disc = self.AdvSKM_embedder(src_feat.detach())
            target_embedding_disc = self.AdvSKM_embedder(trg_feat.detach())
            mmd_loss = - self.mmd_loss(source_embedding_disc, target_embedding_disc)
            mmd_loss.requires_grad = True

            # update discriminator
            self.optimizer_disc.zero_grad()
            mmd_loss.backward()
            self.optimizer_disc.step()

            # calculate source classification loss
            src_cls_loss = self.cross_entropy(src_pred, src_y)

            # domain loss.
            source_embedding_disc = self.AdvSKM_embedder(src_feat)
            target_embedding_disc = self.AdvSKM_embedder(trg_feat)

            mmd_loss_adv = self.mmd_loss(source_embedding_disc, target_embedding_disc)
            mmd_loss_adv.requires_grad = True

            # calculate the total loss
            loss = self.hparams["domain_loss_wt"] * mmd_loss_adv + \
                self.hparams["src_cls_loss_wt"] * src_cls_loss

            # update optimizer
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(), 'MMD_loss': mmd_loss_adv.item(), 'Src_cls_loss': src_cls_loss.item()}
            for key, val in losses.items():
                    avg_meter[key].update(val, 32)

        self.lr_scheduler.step()

class SASA(Algorithm):

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # feature_length for classifier
        configs.features_len = 1
        self.classifier = classifier(configs)
        # feature length for feature extractor
        configs.features_len = 1
        self.feature_extractor = CNN_ATTN(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device


    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)         # extract source features

            # Extract features
            src_feature = self.feature_extractor(src_x)
            tgt_feature = self.feature_extractor(trg_x)

            # source classification loss
            y_pred = self.classifier(src_feature)
            src_cls_loss = self.cross_entropy(y_pred, src_y)

            # MMD loss
            domain_loss_intra = self.mmd_loss(src_struct=src_feature,
                                            tgt_struct=tgt_feature, weight=self.hparams['domain_loss_wt'])

            # total loss
            total_loss = self.hparams['src_cls_loss_wt'] * src_cls_loss + domain_loss_intra

            # remove old gradients
            self.optimizer.zero_grad()
            # calculate gradients
            total_loss.backward()
            # update the weights
            self.optimizer.step()

            losses =  {'Total_loss': total_loss.item(), 'MMD_loss': domain_loss_intra.item(),
                    'Src_cls_loss': src_cls_loss.item()}
            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()
    def mmd_loss(self, src_struct, tgt_struct, weight):
        delta = torch.mean(src_struct - tgt_struct, dim=-2)
        loss_value = torch.norm(delta, 2) * weight
        return loss_value


class CoTMix(Algorithm):
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

         # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Aligment losses
        self.contrastive_loss = NTXentLoss(device, hparams["batch_size"], 0.2, True)
        self.entropy_loss = ConditionalEntropyLoss()
        self.sup_contrastive_loss = SupConLoss(device)

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)         # extract source features

            # ====== Temporal Mixup =====================
            src_dominant, trg_dominant = self.temporal_mixup(src_x, trg_x)

            # ====== Source =====================
            self.optimizer.zero_grad()

            # Src original features
            src_orig_feat = self.feature_extractor(src_x)
            src_orig_logits = self.classifier(src_orig_feat)

            # Target original features
            trg_orig_feat = self.feature_extractor(trg_x)
            trg_orig_logits = self.classifier(trg_orig_feat)

            # -----------  The two main losses
            # Cross-Entropy loss
            src_cls_loss = self.cross_entropy(src_orig_logits, src_y)
            loss = src_cls_loss * round(self.hparams["src_cls_weight"], 2)

            # Target Entropy loss
            trg_entropy_loss = self.entropy_loss(trg_orig_logits)
            loss += trg_entropy_loss * round(self.hparams["trg_entropy_weight"], 2)

            # -----------  Auxiliary losses
            # Extract source-dominant mixup features.
            src_dominant_feat = self.feature_extractor(src_dominant)
            src_dominant_logits = self.classifier(src_dominant_feat)

            # supervised contrastive loss on source domain side
            src_concat = torch.cat([src_orig_logits.unsqueeze(1), src_dominant_logits.unsqueeze(1)], dim=1)
            src_supcon_loss = self.sup_contrastive_loss(src_concat, src_y)
            loss += src_supcon_loss * round(self.hparams["src_supCon_weight"], 2)

            # Extract target-dominant mixup features.
            trg_dominant_feat = self.feature_extractor(trg_dominant)
            trg_dominant_logits = self.classifier(trg_dominant_feat)

            # Unsupervised contrastive loss on target domain side
            trg_con_loss = self.contrastive_loss(trg_orig_logits, trg_dominant_logits)
            loss += trg_con_loss * round(self.hparams["trg_cont_weight"], 2)

            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(),
                    'src_cls_loss': src_cls_loss.item(),
                    'trg_entropy_loss': trg_entropy_loss.item(),
                    'src_supcon_loss': src_supcon_loss.item(),
                    'trg_con_loss': trg_con_loss.item()
                    }
            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()

    def temporal_mixup(self,src_x, trg_x):

        mix_ratio = round(self.hparams["mix_ratio"], 2)
        temporal_shift = self.hparams["temporal_shift"]
        h = temporal_shift // 2  # half

        src_dominant = mix_ratio * src_x + (1 - mix_ratio) * \
                    torch.mean(torch.stack([torch.roll(trg_x, -i, 2) for i in range(-h, h)], 2), 2)

        trg_dominant = mix_ratio * trg_x + (1 - mix_ratio) * \
                    torch.mean(torch.stack([torch.roll(src_x, -i, 2) for i in range(-h, h)], 2), 2)

        return src_dominant, trg_dominant



# Untied Approaches: (MCD)
class MCD(Algorithm):
    """
    Maximum Classifier Discrepancy for Unsupervised Domain Adaptation
    MCD: https://arxiv.org/pdf/1712.02560.pdf
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        self.classifier2 = classifier(configs)

        self.network = nn.Sequential(self.feature_extractor, self.classifier)


        # optimizer and scheduler
        self.optimizer_fe = torch.optim.Adam(
            self.feature_extractor.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
                # optimizer and scheduler
        self.optimizer_c1 = torch.optim.Adam(
            self.classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
                # optimizer and scheduler
        self.optimizer_c2 = torch.optim.Adam(
            self.classifier2.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.lr_scheduler_fe = StepLR(self.optimizer_fe, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        self.lr_scheduler_c1 = StepLR(self.optimizer_c1, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        self.lr_scheduler_c2 = StepLR(self.optimizer_c2, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Aligment losses
        self.mmd_loss = MMD_loss()

    def update(self, src_loader, trg_loader, avg_meter, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None

        for epoch in range(1, self.hparams["num_epochs"] + 1):

            # source pretraining loop 
            self.pretrain_epoch(src_loader, avg_meter)

            # training loop 
            self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())


            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        last_model = self.network.state_dict()

        return last_model, best_model

    def pretrain_epoch(self, src_loader,avg_meter):
        for src_x, src_y in src_loader:
            src_x, src_y = src_x.to(self.device), src_y.to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred1 = self.classifier(src_feat)
            src_pred2 = self.classifier2(src_feat)

            src_cls_loss1 = self.cross_entropy(src_pred1, src_y)
            src_cls_loss2 = self.cross_entropy(src_pred2, src_y)

            loss = src_cls_loss1 + src_cls_loss2

            self.optimizer_c1.zero_grad()
            self.optimizer_c2.zero_grad()
            self.optimizer_fe.zero_grad()

            loss.backward()

            self.optimizer_c1.step()
            self.optimizer_c2.step()
            self.optimizer_fe.step()


            losses = {'Src_cls_loss': loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)           # extract source features


            # extract source features
            src_feat = self.feature_extractor(src_x)
            src_pred1 = self.classifier(src_feat)
            src_pred2 = self.classifier2(src_feat)

            # source losses
            src_cls_loss1 = self.cross_entropy(src_pred1, src_y)
            src_cls_loss2 = self.cross_entropy(src_pred2, src_y)
            loss_s = src_cls_loss1 + src_cls_loss2


            # Freeze the feature extractor
            for k, v in self.feature_extractor.named_parameters():
                v.requires_grad = False
            # update C1 and C2 to maximize their difference on target sample
            trg_feat = self.feature_extractor(trg_x)
            trg_pred1 = self.classifier(trg_feat.detach())
            trg_pred2 = self.classifier2(trg_feat.detach())


            loss_dis = self.discrepancy(trg_pred1, trg_pred2)

            loss = loss_s - loss_dis

            loss.backward()
            self.optimizer_c1.step()
            self.optimizer_c2.step()

            self.optimizer_c1.zero_grad()
            self.optimizer_c2.zero_grad()
            self.optimizer_fe.zero_grad()

            # Freeze the classifiers
            for k, v in self.classifier.named_parameters():
                v.requires_grad = False
            for k, v in self.classifier2.named_parameters():
                v.requires_grad = False
                        # Freeze the feature extractor
            for k, v in self.feature_extractor.named_parameters():
                v.requires_grad = True
            # update feature extractor to minimize the discrepaqncy on target samples
            trg_feat = self.feature_extractor(trg_x)
            trg_pred1 = self.classifier(trg_feat)
            trg_pred2 = self.classifier2(trg_feat)


            loss_dis_t = self.discrepancy(trg_pred1, trg_pred2)
            domain_loss = self.hparams["domain_loss_wt"] * loss_dis_t

            domain_loss.backward()
            self.optimizer_fe.step()

            self.optimizer_fe.zero_grad()
            self.optimizer_c1.zero_grad()
            self.optimizer_c2.zero_grad()


            losses =  {'Total_loss': loss.item(), 'MMD_loss': domain_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler_fe.step()
        self.lr_scheduler_c1.step()
        self.lr_scheduler_c2.step()

    def discrepancy(self, out1, out2):

        return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))


class NO_ADAPT_WITH_AE(Algorithm):
    """
    Lower bound: train on source and test on target.
    With an autoencoder to add reconstruction losses for both src_x and trg_x.
    """
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # Define the decoder (inverse of feature_extractor)

        encoder_name = str(type(self.feature_extractor)).split('.')[-1][:-2]
        self.decoder = globals()[encoder_name + '_Decoder'](configs)

        # Update the network to include both feature extractor and decoder
        self.autoencoder = nn.Sequential(self.feature_extractor, self.decoder)

        # Optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            list(self.network.parameters()) + list(self.decoder.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # Hyperparameters
        self.hparams = hparams

        # Device
        self.device = device

        # Reconstruction loss (Mean Squared Error for autoencoder)
        self.reconstruction_loss = nn.MSELoss()

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):
        # Set the model to training mode
        self.train()

        # Iterate through both source and target data loaders
        for (src_x, src_y), (trg_x, _) in zip(src_loader, trg_loader):
            # Move data to the appropriate device
            src_x, src_y = src_x.to(self.device), src_y.to(self.device)
            trg_x = trg_x.to(self.device)

            # Forward pass for the source domain
            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)
            src_reconstructed = self.decoder(src_feat)

            # Forward pass for the target domain
            trg_feat = self.feature_extractor(trg_x)
            trg_reconstructed = self.decoder(trg_feat)

            # Compute classification loss for the source domain
            src_cls_loss = self.cross_entropy(src_pred, src_y)

            # Compute reconstruction loss for both source and target
            src_reconstruction_loss = self.reconstruction_loss(src_reconstructed, src_x)
            trg_reconstruction_loss = self.reconstruction_loss(trg_reconstructed, trg_x)

            # Total loss is the sum of classification and reconstruction losses
            loss = ((1 - self.hparams["lambda_reconstruct"]) * src_cls_loss
                    + self.hparams["lambda_reconstruct"] * (src_reconstruction_loss + trg_reconstruction_loss))

            # Backpropagation and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Log losses
            losses = {
                'Src_cls_loss': src_cls_loss.item(),
                'Src_reconstruct_loss': src_reconstruction_loss.item(),
                'Trg_reconstruct_loss': trg_reconstruction_loss.item(),
            }

            for key, val in losses.items():
                avg_meter[key].update(val, src_x.size(0))

        # Step the learning rate scheduler
        self.lr_scheduler.step()

class ADDA_Discriminator(nn.Module):
    """ADDA 用的域判别器，输入为 encoder 提取的特征"""
    def __init__(self, configs, hparams):
        super().__init__()
        input_dim = configs.features_len * configs.final_out_channels
        hidden_dim = hparams.get("disc_hid_dim", 500)
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)   # 2 类：源域/目标域
        )

    def forward(self, x):
        return self.layer(x.flatten(1))


class ADDA(Algorithm):
    """
    Adversarial Discriminative Domain Adaptation (Tzeng et al., 2017)
    适配到 AdaTime 框架的时序版本
    两阶段：
      1) 0 ~ pretrain_epochs:        监督训练 src_encoder + classifier
      2) pretrain_epochs ~ num_epochs: 冻结 src_encoder/classifier，
                                       对抗训练 tgt_encoder vs discriminator
    推理时使用 tgt_encoder + classifier
    """
    def __init__(self, backbone, configs, hparams, device):
        # 父类会创建 self.feature_extractor / self.classifier / self.network
        super().__init__(configs, backbone)

        self.device = device
        self.hparams = hparams

        # 复用父类创建的 feature_extractor 作为 src_encoder
        self.src_encoder = self.feature_extractor

        # 目标域 encoder（结构同 src_encoder，权重在 adapt 阶段从 src_encoder 拷贝）
        self.tgt_encoder = backbone(configs)

        # 域判别器
        self.discriminator = ADDA_Discriminator(configs, hparams).to(device)

        # ----- 优化器 -----
        # 阶段1：源域分类
        self.src_optimizer = torch.optim.Adam(
            list(self.src_encoder.parameters()) + list(self.classifier.parameters()),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["src_weight_decay"]
        )
        # 阶段2：tgt encoder（对抗）
        self.tgt_optimizer = torch.optim.Adam(
            self.tgt_encoder.parameters(),
            lr=hparams["learning_rate"],
            betas=(0.5, 0.9),
            weight_decay=hparams["weight_decay"]
        )
        # 阶段2：discriminator
        self.disc_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=hparams["learning_rate"],
            betas=(0.5, 0.9),
            weight_decay=hparams["weight_decay"]
        )

        # 阶段1 lr 调度
        self.lr_scheduler = StepLR(self.src_optimizer,
                                   step_size=hparams['step_size'],
                                   gamma=hparams['lr_decay'])

        # 损失
        self.cls_criterion = nn.CrossEntropyLoss()
        self.adv_criterion = nn.CrossEntropyLoss()

        # 阶段控制
        self.pretrain_epochs = hparams["pretrain_epochs"]
        self.current_epoch = 0
        self.is_pretrained = False

    # ===================================================================
    # AdaTime 的入口：按 epoch 调用
    # ===================================================================
    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):
        # 基类 update 里会设置 self.current_epoch = epoch - 1
        if self.current_epoch < self.pretrain_epochs:
            self._pretrain_epoch(src_loader, avg_meter)
            self.lr_scheduler.step()
        else:
            # 第一次进入 adapt 阶段时，把 src_encoder 权重拷给 tgt_encoder
            if not self.is_pretrained:
                self.tgt_encoder.load_state_dict(self.src_encoder.state_dict())
                self.is_pretrained = True
                # 关键：让 AdaTime 框架在推理 / 保存 checkpoint 时用 tgt_encoder
                self.feature_extractor = self.tgt_encoder
                self.network = nn.Sequential(self.tgt_encoder, self.classifier)

            self._adapt_epoch(src_loader, trg_loader, avg_meter, epoch)

    # ===================================================================
    # 阶段 1：源域监督预训练
    # ===================================================================
    def _pretrain_epoch(self, src_loader, avg_meter):
        self.src_encoder.train()
        self.classifier.train()

        for src_x, src_y in src_loader:
            src_x, src_y = src_x.to(self.device), src_y.to(self.device)

            src_feat = self.src_encoder(src_x)
            src_pred = self.classifier(src_feat)
            loss = self.cls_criterion(src_pred, src_y)

            self.src_optimizer.zero_grad()
            loss.backward()
            self.src_optimizer.step()

            avg_meter['Src_cls_loss'].update(loss.item(), src_x.size(0))

    # ===================================================================
    # 阶段 2：对抗自适应
    # ===================================================================
    def _adapt_epoch(self, src_loader, trg_loader, avg_meter, epoch):
        # 冻结源 encoder 和 classifier
        self.src_encoder.eval()
        self.classifier.eval()

        joint_loader = enumerate(zip(src_loader, itertools.cycle(trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = (src_x.to(self.device),
                                   src_y.to(self.device),
                                   trg_x.to(self.device))

            self.tgt_encoder.train()
            self.discriminator.train()

            bs_src = src_x.size(0)
            bs_trg = trg_x.size(0)

            # -------- (a) 训练 discriminator --------
            with torch.no_grad():
                src_feat = self.src_encoder(src_x).flatten(1)
                tgt_feat_d = self.tgt_encoder(trg_x).flatten(1)

            feat_concat = torch.cat([src_feat, tgt_feat_d], dim=0)
            pred_concat = self.discriminator(feat_concat)

            label_src = torch.ones(bs_src,  dtype=torch.long, device=self.device)
            label_tgt = torch.zeros(bs_trg, dtype=torch.long, device=self.device)
            label_concat = torch.cat([label_src, label_tgt], dim=0)

            d_loss = self.adv_criterion(pred_concat, label_concat)

            self.disc_optimizer.zero_grad()
            d_loss.backward()
            self.disc_optimizer.step()

            d_acc = (pred_concat.argmax(1) == label_concat).float().mean().item()

            # -------- (b) 训练 target encoder (骗过 D) --------
            tgt_feat = self.tgt_encoder(trg_x).flatten(1)
            pred_tgt = self.discriminator(tgt_feat)
            # 翻转标签：希望 D 把 target 判为 source(1)
            label_flip = torch.ones(bs_trg, dtype=torch.long, device=self.device)
            g_loss = self.adv_criterion(pred_tgt, label_flip)

            self.tgt_optimizer.zero_grad()
            g_loss.backward()
            self.tgt_optimizer.step()

            # -------- (c) 监控用：源域分类 loss（不反传）--------
            with torch.no_grad():
                src_pred = self.classifier(self.src_encoder(src_x))
                src_cls_loss_val = self.cls_criterion(src_pred, src_y).item()

            losses = {
                'Src_cls_loss': src_cls_loss_val,   # 为了 best_model 选择机制
                'Disc_loss':    d_loss.item(),
                'Gen_loss':     g_loss.item(),
                'Disc_acc':     d_acc,
            }
            for key, val in losses.items():
                avg_meter[key].update(val, 32)