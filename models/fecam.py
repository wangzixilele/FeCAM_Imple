
import logging
import numpy as np
from tqdm import tqdm
import torch

from torch import nn
from torch import optim
from torch import linalg as LA
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from models.base import BaseLearner
from utils.inc_net import CosineIncrementalNet, IncrementalNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy
from torchvision import datasets, transforms
from utils.autoaugment import CIFAR10Policy
from utils.maha_utils import compute_common_cov, compute_new_common_cov, compute_new_cov
from resnet import resnet18, resnet50, resnet101


EPSILON = 1e-8


class FeCAM(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = CosineIncrementalNet(args, False)
        self._protos = []
        self._init_protos = []
        self._common_cov = None
        self._cov_mat = []
        self._diag_mat = []
        self._common_cov_shrink = None
        self._cov_mat_shrink = []
        self._norm_cov_mat = []

    def after_task(self):
        self._known_classes = self._total_classes
        # if self._cur_task == 0:
        #     self.save_checkpoint("{}_{}_{}_{}".format(self.args["dataset"],self.args["model_name"],self.args["init_cls"],self.args["increment"]))
        
    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1
        if self.args['dataset'] == "cifar100":
            self.data_manager._train_trsf = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=63/255),
                CIFAR10Policy(),
                transforms.ToTensor(),
            ]
        elif self.args['dataset'] == "tinyimagenet200":
            self.data_manager._train_trsf = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.ToPILImage()
            ]
        elif self.args['dataset'] == "imagenet100":
            self.data_manager._train_trsf = [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.ToPILImage()
            ]

        self._total_classes = self._known_classes + \
            data_manager.get_task_size(self._cur_task)

        self._network.update_fc(self._total_classes, self._cur_task)
        self._network_module_ptr = self._network
        logging.info(
            'Learning on {}-{}'.format(self._known_classes, self._total_classes))

        if self._cur_task > 0:   # Freezing the network
            for p in self._network.convnet.parameters():
                p.requires_grad = False
        
        self.shot = None

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train', shot=self.shot)  

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=self.args["num_workers"], pin_memory=True)
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=self.args["num_workers"])

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        self._train(self.train_loader, self.test_loader)

        


    def _train(self, train_loader, test_loader):
        resume = False  # set resume=True to use saved checkpoints after first task
        if self._cur_task == 0:
            if resume:
                self._network.load_state_dict(torch.load("{}_{}_{}_{}_{}.pkl".format(self.args["dataset"],self.args["model_name"],self.args["init_cls"],self.args["increment"],self._cur_task))["model_state_dict"])
            self._network.to(self._device)
            if hasattr(self._network, "module"):
                self._network_module_ptr = self._network.module
            if not resume:
                self._epoch_num = self.args["init_epochs"]
                optimizer = optim.SGD(filter(lambda p: p.requires_grad, self._network.parameters(
                )), momentum=0.9, lr=self.args["init_lr"], weight_decay=self.args["init_weight_decay"])
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer, T_max=self.args["init_epochs"])
                self._train_function(train_loader, test_loader, optimizer, scheduler)
            self._build_base_protos()
            self._build_protos()
            if self.args["full_cov"]:
                if self.args["per_class"]:
                    compute_new_cov(self)
                    if self.args["shrink"]:  # we apply covariance shrinkage 2 times to obtain better estimates of matrices
                        for cov in self._cov_mat:
                            self._cov_mat_shrink.append(self.shrink_cov(cov))
                    if self.args["norm_cov"]:
                        self._norm_cov_mat = self.normalize_cov()
                else:
                    self._common_cov = compute_common_cov(train_loader, self)
            elif self.args["diagonal"]:
                if self.args["per_class"]:
                    compute_new_cov(self)
                    for cov in self._cov_mat:
                        self._cov_mat_shrink.append(self.shrink_cov(cov))
                    for cov in self._cov_mat_shrink:
                        cov = self.normalize_cov2(cov)
                        self._diag_mat.append(self.diagonalization(cov))
        else:
            self._cov_mat_shrink, self._norm_cov_mat, self._diag_mat = [], [], []
            self._network.to(self._device)
            if hasattr(self._network, "module"):
                self._network_module_ptr = self._network.module
            self._build_protos()
            self._update_fc()
            if self.args["full_cov"]:
                if self.args["per_class"]:
                    compute_new_cov(self)
                    if self.args["shrink"]:
                        for cov in self._cov_mat:
                            self._cov_mat_shrink.append(self.shrink_cov(cov))
                    if self.args["norm_cov"]:
                        self._norm_cov_mat = self.normalize_cov()
                else:
                    self._common_cov = compute_new_common_cov(train_loader, self)
            elif self.args["diagonal"]:
                if self.args["per_class"]:
                    compute_new_cov(self)
                    for cov in self._cov_mat:
                        self._cov_mat_shrink.append(self.shrink_cov(cov))
                    for cov in self._cov_mat_shrink:
                        cov = self.normalize_cov2(cov)
                        self._diag_mat.append(self.diagonalization(cov))

    
    def _build_base_protos(self):
        for class_idx in range(self._known_classes, self._total_classes):
            class_mean = self._network.fc.weight.data[class_idx]
            self._init_protos.append(class_mean)

    def _build_protos(self):
        with torch.no_grad():
            for class_idx in range(self._known_classes, self._total_classes):
                data, targets, idx_dataset = self.data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                    mode='test', shot=self.shot, ret_data=True)
                idx_loader = DataLoader(idx_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4)
                vectors, _ = self._extract_vectors(idx_loader)
                class_mean = np.mean(vectors, axis=0)
                self._protos.append(torch.tensor(class_mean).to(self._device))

    def _update_fc(self):
        self._network.fc.fc2.weight.data = torch.stack(self._protos[-self.args["increment"]:], dim=0).to(self._device)  # for cosine incremental fc layer

    
    def _train_function(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self._epoch_num))
        # source_index = self.args["source_index"]
        # class_num = 126
        # source_model_path = ["/public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/domainNet-126/20240429-2246-OH_clipart_ce_singe_gpu_resnet50_best_param.pth",
        #                     "/public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/domainNet-126/20240429-2246-OH_painting_ce_singe_gpu_resnet50_best_param.pth",
        #                     "/public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/domainNet-126/20240429-2246-OH_real_ce_singe_gpu_resnet50_best_param.pth",
        #                     "/public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/domainNet-126/20240429-2246-OH_sketch_ce_singe_gpu_resnet50_best_param.pth"]
        
        # class_num = 31
        # source_model_path = ["/public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/office31/20240427-2132-OH_amazon_ce_singe_gpu_resnet50_best.pkl",
        #                      "/public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/office31/20240427-2133-OH_dslr_ce_singe_gpu_resnet50_best.pkl",
        #                      "/public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/office31/20240427-2211-OH_webcam_ce_singe_gpu_resnet50_best.pkl",

        # class_num = 65
        # source_model_path = ["/public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/office_home/20240414-0211-OH_Art_ce_singe_gpu_resnet50_best_param.pth",
        #                      "/public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/office_home/20240322-1652-OH_Clipart_ce_singe_gpu_resnet50_best_param.pth",
        #                      "/public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/office_home/20240414-0239-OH_Product_ce_singe_gpu_resnet50_best_param.pth",
        #                      "/public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/office_home/20240414-0307-OH_World_ce_singe_gpu_resnet50_best_param.pth"]
        
        # source_model = resnet50(pretrained=True)
        # source_model.fc = nn.Linear(2048, class_num)#need to change
        # source_model.load_state_dict(torch.load(source_model_path[source_index]))
        # source_model = source_model.to(self._device)
        # source_model.eval()

        for _, epoch in enumerate(prog_bar):
            if self._cur_task == 0:
                self._network.train()
            else:
                self._network.eval()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(
                    self._device, non_blocking=True), targets.to(self._device, non_blocking=True)
                if self._cur_task ==0:
                    logits = self._network(inputs)['logits']
                else:
                    logits = self._network_module_ptr.fc(inputs)['logits']

                # source_logits = source_model(inputs)[1]
                # pseduo_targets = torch.softmax(source_logits, dim=1)
                # pseduo_targets = pseduo_targets[:,:self._total_classes]
                # pseduo_targets = torch.argmax(pseduo_targets, dim=1)
                
                loss = F.cross_entropy(logits,targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(
                correct)*100 / total, decimals=2)
            if epoch % 5 != 0:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                    self._cur_task, epoch+1, self._epoch_num, losses/len(train_loader), train_acc)
            else:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                    self._cur_task, epoch+1, self._epoch_num, losses/len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)
            logging.info(info)
