"""
Author: Rex Geng
This file inits all nessceary configurations given usr specificed yaml file. The configuration includes:

- device
- learning rate scheduler
- preprocessing (nn transforms)
- loss function (crossEntropy, weighted cross entropy, mse ...)
- optimizer (sgd, adam ... )
- dataset
- dataloader
- neural network model
- trainer class (backprop, rnn, autoencoder ...)
"""

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nnd  # existing torch nn package
from torchvision import transforms as tfd  # existing transforms

from nn import dataset as datasetc
from nn import loss as lossc
from nn import lr_scheduler as lrsc
from nn import models as modelsc
from nn import optimizer as optimizerc
from nn import quantization
from nn import transforms as tfc  # customized transforms
from nn.trainer import __TRAINER__
from nn.utils import ImbalancedDatasetSampler


def get_device(device_usr_configs):
    device = 'cuda' if device_usr_configs.use_gpu else 'cpu'
    if device == 'cuda':
        device = "{}:{}".format(device, device_usr_configs.gpu_id)
    return device


def get_lr_scheduler(optimizer, lr_scheduler_usr_configs):
    lr_class = lrsc.__SCHEDULER__.get(lr_scheduler_usr_configs.name)
    if lr_class is None:
        raise NotImplementedError('lr scheduler not recognized {}'.format(lr_scheduler_usr_configs.name))

    scheduler = lr_class(
        optimizer, **lr_scheduler_usr_configs.init_args.__dict__
    )
    return scheduler


def __get_preprocessing__(preprocessing_usr_configs):
    transforms = []
    for t in preprocessing_usr_configs:
        tclass = tfc.__TRANSFORMS__.get(t.name)
        if tclass is None:
            raise NotImplementedError('transforms not recognized {}'.format(t.name))
        tobj = tclass(**t.init_args.__dict__)
        transforms.append(tobj)
    return tfd.Compose(transforms)


def get_loss_func(loss_func_usr_configs, weight=None):
    if loss_func_usr_configs.name == 'cel':
        criterion = nnd.CrossEntropyLoss()
    elif loss_func_usr_configs.name == 'wcel':
        criterion = nnd.CrossEntropyLoss(weight=weight)
    elif loss_func_usr_configs.name == 'mse':
        criterion = nnd.MSELoss()
    else:
        return NotImplementedError('loss function name not recognized')

    loss_class = lossc.__LOSS__.get(loss_func_usr_configs.name)
    if loss_class is None:
        raise NotImplementedError('loss function name not recognized {}'.format(loss_func_usr_configs.name))

    if weight is not None:
        criterion = loss_class(weight=weight)
    else:
        criterion = loss_class()
    return criterion


def get_optimizer(model, optimizer_usr_configs):
    optimizer_class = optimizerc.__OPTIMIZER__.get(optimizer_usr_configs.name)
    if optimizer_class is None:
        raise NotImplementedError('optimizer name not recognized {}'.format(optimizer_usr_configs.name))

    optimizer = optimizer_class(
        model.parameters(), **optimizer_usr_configs.init_args.__dict__
    )
    return optimizer


def get_data_loader(trainer_usr_configs, train_dataset=None, validation_dataset=None, test_dataset=None):
    train_data_loader = None
    validation_data_loader = None
    test_data_loader = None

    if train_dataset is not None:
        if trainer_usr_configs.use_train_weighted_sampler:
            train_sampler = ImbalancedDatasetSampler(train_dataset)
            trkwargs = dict(
                batch_size=trainer_usr_configs.batch_size,
                num_workers=trainer_usr_configs.num_worker, pin_memory=True,
                drop_last=trainer_usr_configs.drop_last_batch,
                sampler=train_sampler
            )
        else:
            trkwargs = dict(
                batch_size=trainer_usr_configs.batch_size,
                num_workers=trainer_usr_configs.num_worker, pin_memory=True,
                drop_last=trainer_usr_configs.drop_last_batch,
                shuffle=True
            )

        train_data_loader = torch.utils.data.DataLoader(train_dataset, **trkwargs)

    if validation_dataset is not None:
        validation_data_loader = torch.utils.data.DataLoader(
            validation_dataset, batch_size=trainer_usr_configs.batch_size, shuffle=False,
            num_workers=trainer_usr_configs.num_worker, pin_memory=True,
            drop_last=trainer_usr_configs.drop_last_batch
        )

    if test_dataset is not None:
        if trainer_usr_configs.use_test_weighted_sampler:
            test_sampler = ImbalancedDatasetSampler(test_dataset)
            trkwargs = dict(
                batch_size=trainer_usr_configs.batch_size,
                num_workers=trainer_usr_configs.num_worker, pin_memory=True,
                drop_last=trainer_usr_configs.drop_last_batch,
                sampler=test_sampler
            )
        else:
            trkwargs = dict(
                batch_size=trainer_usr_configs.batch_size,
                num_workers=trainer_usr_configs.num_worker, pin_memory=True,
                drop_last=trainer_usr_configs.drop_last_batch,
                shuffle=False
            )
        test_data_loader = torch.utils.data.DataLoader(test_dataset, **trkwargs)

    return train_data_loader, validation_data_loader, test_data_loader


def get_model(model_usr_configs):
    model_cls = modelsc.__MODELS__.get(model_usr_configs.name)
    if model_cls is None:
        raise NotImplementedError('model not implemented')

    model_obj = model_cls(**model_usr_configs.init_args.__dict__)

    if model_usr_configs.quant_model:
        model_obj = quantization.quantize_model(model_obj, model_usr_configs)

    return model_obj


def get_dataset(dataset_usr_configs):
    dataset_cls = datasetc.__DATASET__.get(dataset_usr_configs.name)

    if hasattr(dataset_usr_configs, 'base_transforms'):
        base_transforms = __get_preprocessing__(dataset_usr_configs.base_transforms)
    else:
        base_transforms = tfd.Compose([])

    if hasattr(dataset_usr_configs, 'aug_transforms'):
        aug_transforms = __get_preprocessing__(dataset_usr_configs.aug_transforms)
    else:
        aug_transforms = tfd.Compose([])

    if dataset_usr_configs.aug:
        total_trainset = dataset_cls(
            transform=aug_transforms,
            train=True,
            **dataset_usr_configs.trainset.__dict__
        )
    else:
        total_trainset = dataset_cls(
            transform=base_transforms,
            train=True,
            **dataset_usr_configs.trainset.__dict__
        )

    testset = dataset_cls(
        transform=base_transforms,
        train=False,
        **dataset_usr_configs.testset.__dict__
    )

    train_size = int(dataset_usr_configs.train_valid_split * len(total_trainset))
    validation_size = len(total_trainset) - train_size
    trainset, validationset = torch.utils.data.random_split(total_trainset, [train_size, validation_size])

    if dataset_usr_configs.use_validation:
        print('splitting validation dataset')
        print('using {} for training, using {} for validation'.format(train_size, validation_size))
        validationset.preprocessing = base_transforms
        return trainset, validationset, testset
    else:
        return total_trainset, None, testset


def get_trainer(trainer_usr_configs):
    trainer_cls = __TRAINER__.get(trainer_usr_configs.trainer)
    if trainer_cls is None:
        raise NotImplementedError('trainer cls not implemented')
    return trainer_cls


class TrainerConfigs:
    def __init__(
            self,
            train_loader=None,
            validation_loader=None,
            test_loader=None,
            model=None,
            optimizer=None,
            criterion=None,
            device=None,
            lr_scheduler=None,
            result_dir=None,
            num_epoch=0,
            resume_from_best=False,
            model_src_path=None,
            batch_size=0,
            log_update_feq=0,
            usr=None
    ):
        self.usr = usr
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.lr_scheduler = lr_scheduler

        self.result_dir = result_dir
        self.num_epoch = num_epoch
        self.resume_from_best = resume_from_best
        self.model_src_path = model_src_path
        self.batch_size = batch_size
        self.log_update_feq = log_update_feq

    def __setup__(self, usr):
        self.usr = usr

        self.model = get_model(usr.model)
        self.optimizer = get_optimizer(self.model, usr.optimizer)
        self.lr_scheduler = get_lr_scheduler(self.optimizer, usr.lr_scheduler)

        if usr.lr_scheduler == 'rdp':
            self.use_loss_metric = True
        else:
            self.use_loss_metric = False

        self.device = get_device(usr.device)
        if self.device != 'cpu':
            if usr.device.parallel:
                self.model = torch.nn.DataParallel(self.model)
            cudnn.benchmark = True

        trainset, valiationset, testset = get_dataset(usr.dataset)
        ret = get_data_loader(usr.train, trainset, valiationset, testset)
        self.trainset = trainset
        self.validationset = valiationset
        self.testset = testset

        self.train_loader = ret[0]
        self.validation_loader = ret[1]
        self.test_loader = ret[2]
        self.criterion = get_loss_func(
            usr.loss_func, weight=trainset.cls_weight if hasattr(trainset, 'cls_weight') else None
        )

        self.batch_size = usr.train.batch_size
        self.num_epoch = usr.train.num_epoch
        self.result_dir = usr.train.result_dir
        self.model_src_path = usr.train.model_src_path
        self.resume_from_best = usr.train.resume_from_best
        self.log_update_feq = usr.train.print_freq
        self.eval_model = not usr.train.train_model

        self.metric = usr.train.save_model_by
        torch.manual_seed(usr.seed)
