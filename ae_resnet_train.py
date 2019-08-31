import os
import torch
import warnings

import torch.nn as nn
import torch.optim as optim
import numpy as np
import utils as utility

import torchvision.transforms as torch_transforms
import torchvision.models as torch_models
import dataloaders.pascal_voc2012 as voc12
import nets.ae_resnet as ae_resnet


def _main(args):
    warnings.filterwarnings("ignore")

    # torch.autograd.set_detect_anomaly(True)

    #### Preparing Dataset ####
    data_dir = './datasets/pascal_voc2012/train/1'
    labels_address = './datasets/pascal_voc2012/data_labels.txt'
    train_data_transform = torch_transforms.Compose([
        torch_transforms.Resize((248, 248))
    ])

    train_obj_area_threshold = 0.
    dataloader = voc12.loader(data_dir,
                              data_transform=train_data_transform,
                              obj_area_threshold=train_obj_area_threshold)
    # data_loader = [1, 2]

    #### Preparing Pytorch ####
    device = args.device
    assert (device in ['cpu', 'multi']) or (len(device.split(':')) == 2 and device.split(':')[
        0] == 'cuda' and int(device.split(':')[1]) < torch.cuda.device_count()), 'Uknown device: {}'.format(device)
    torch.manual_seed(0)
    if args.device != 'multi':
        device = torch.device(args.device)
    if args.gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    #### Preparing EA layers ####
    ae_layers_mode = args.ae_layers
    print('##### ae_layers_mode: {}'.format(ae_layers_mode))
    ae_layers_mode = ae_layers_mode.split(';')
    ae_layers = {
        'conv1': ae_layers_mode[0],
        'layer1': [layer_mode for layer_mode in ae_layers_mode[1].split(',')],
        'layer2': [layer_mode for layer_mode in ae_layers_mode[2].split(',')],
        'layer3': [layer_mode for layer_mode in ae_layers_mode[3].split(',')],
        'layer4': [layer_mode for layer_mode in ae_layers_mode[4].split(',')]
    }
    ae_rate = args.ae_rate
    print('**** ae_layers: {}'.format(ae_layers))
    print('**** ae_rate: {}'.format(ae_rate))

    #### Constructing Model ####
    pretrained = args.pretrained
    model = None
    print('#### pretrained: {}'.format(pretrained))
    if args.resnet_type == 'resnet18':
        model = ae_resnet.resnet18(
            pretrained=pretrained, ae_layers=ae_layers, ae_alpha=ae_rate)
    elif args.resnet_type == 'resnet34':
        model = ae_resnet.resnet34(
            pretrained=pretrained, ae_layers=ae_layers, ae_alpha=ae_rate)
    elif args.resnet_type == 'resnet50':
        model = ae_resnet.resnet50(
            pretrained=pretrained, ae_layers=ae_layers, ae_alpha=ae_rate)
    elif args.resnet_type == 'resnet101':
        model = ae_resnet.resnet101(
            pretrained=pretrained, ae_layers=ae_layers, ae_alpha=ae_rate)
    elif args.resnet_type == 'resnet152':
        model = ae_resnet.resnet152(
            pretrained=pretrained, ae_layers=ae_layers, ae_alpha=ae_rate)
    elif args.resnet_type == 'resnext50_32x4d':
        model = ae_resnet.resnet18(
            pretrained=pretrained, ae_layers=ae_layers, ae_alpha=ae_rate)
    elif args.resnet_type == 'resnext101_32x8d':
        model = ae_resnet.resnext101_32x8d(
            pretrained=pretrained, ae_layers=ae_layers, ae_alpha=ae_rate)

    model.fc = nn.Linear(model.fc.in_features, 12)

    if args.gpu and torch.cuda.is_available():
        if device == 'multi':
            model = nn.DataParallel(model)
        else:
            model = model.cuda(device=device)

    #### Constructing Criterion ####
    criterion = nn.CrossEntropyLoss()

    #### Constructing Optimizer ####
    optimizer = None
    if args.optimization == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=args.lr, momentum=args.momentum)
    elif args.optimization == 'adam':
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    #### Training Parameters ####
    start_epoch, num_epoch = (args.start_epoch, args.epochs)
    batch_size = args.batch_size
    num_workers = args.num_workers
    check_counter = 10

    #### Reports Address ####
    reports_root = './reports'
    analysis_num = args.analysis
    reports_path = '{}/{}'.format(reports_root, analysis_num)
    saving_model_path = '{}/models'.format(reports_path)
    model_name = 'ae_resnet_{}_{}'.format(start_epoch, start_epoch+num_epoch)

    utility.mkdir(reports_path, 'models', forced_remove=False)

    #### Training Model ####
    ae_resnet.train(
        ae_resnet=model,
        train_data=dataloader,
        optimizer=optimizer,
        criterion=criterion,
        num_epoch=num_epoch,
        start_epoch=start_epoch,
        batch_size=batch_size,
        num_workers=num_workers,
        check_counter=check_counter,
        gpu=args.gpu and torch.cuda.is_available(),
        report_path=reports_path,
        saving_model_every_epoch=True,
        device=device
    )

    #### Saving Model ####
    ae_resnet.save(saving_model_path, model_name, model, optimizer=optimizer)


if __name__ == "__main__":
    args = utility.get_args()
    _main(args)
