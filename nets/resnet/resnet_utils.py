import gc
import os
import torch
import time

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import utils as utility

from torch.autograd import Variable as V
from torch.utils.data import DataLoader

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

def resnet_save( file_path, file_name, resnet, optimizer=None ):
    state_dict = {
        'net_arch' : 'resnet',
        'model' : resnet.state_dict(),
    }
    if optimizer is not None:
        state_dict[ 'optimizer' ] = optimizer.state_dict()

    torch.save( state_dict, '{}/{}.pth'.format(file_path,file_name) )

def resnet_load(file_path, file_name, model, optimizer=None):
    check_points = torch.load('{}/{}.pth'.format(file_path,file_name))
    keys = check_points.keys()


    assert ('net_arch' in keys) and ('model' in keys), 'Cannot read this file in address : {}/{}.pth'.format(file_path,file_name)
    assert check_points['net_arch']=='resnet', 'This file model architecture is not \'resnet\''
    model.load_state_dict( check_points['model'] )
    if optimizer is not None:
        optimizer.load_state_dict(check_points['optimizer'])
    return model, optimizer

def resnet_accuracy( predicted_output, expected_output ):
    predicted_category = torch.argmax( predicted_output, dim=1 )
    expected_category = torch.argmax( expected_output, dim=1 )
    tp_results = predicted_category==expected_category

    return int( torch.sum(tp_results) ) / predicted_category.size()[0]

def resnet_train(
                 resnet,
                 train_data,
                 optimizer,
                 criterion,
                 report_path,
                 device,
                 curriculum_layers,
                 curriculum_rate=0.,
                 num_epoch=1,
                 start_epoch=0, 
                 batch_size=2,
                 num_workers = 1,
                 check_counter=20,
                 gpu=False,
                 saving_model_every_epoch=False):

    utility.mkdir( report_path, 'train_batches_size' )
    utility.mkdir( report_path, 'train_losses' )
    utility.mkdir( report_path, 'train_accuracies' )
    utility.mkdir( report_path, 'models' )

    for epoch in range( start_epoch, start_epoch+num_epoch ):

        data_loader = DataLoader( train_data,
                                batch_size=batch_size,
                                shuffle=False,
                                pin_memory= gpu and torch.cuda.is_available(),
                                num_workers=num_workers)
        
        batches_size = list()
        losses = list()
        accuracies = list()

        curr_loss = 0
        for ix,(X, Y) in enumerate( data_loader ):
            X, Y = V(X), V(Y)
            
            if gpu:
                if device=='multi':
                    X, Y = nn.DataParallel(X), nn.DataParallel(Y)
                else:
                    X, Y = X.cuda(device=device), Y.cuda(device=device)

            output = resnet( (X,None), curriculum_layers=curriculum_layers, curriculum_rate=curriculum_rate )

            target = Y
            if isinstance( criterion, nn.CrossEntropyLoss ):
                target = torch.argmax( target, dim=1 )

            loss = criterion( output, target )
            acc = resnet_accuracy(output, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prev_loss = curr_loss
            curr_loss = loss.item()

            batches_size.append( output.size()[0] )
            losses.append( curr_loss )
            accuracies.append( acc )

            print( 'epoch=%d, batch=%d(x%d), prev_loss=%.3f, curr_loss=%.3f, delta=%.3f, acc=%.3f%%' % (
                                                                                                        epoch,
                                                                                                        ix,
                                                                                                        output.size()[0],
                                                                                                        prev_loss,
                                                                                                        curr_loss,
                                                                                                        curr_loss-prev_loss,
                                                                                                        acc*100
                                                                                                    ) )
            if ix%check_counter==(check_counter-1):
                # print()
                pass
            
            del X, Y, output       
            torch.cuda.empty_cache()
            gc.collect()

        torch.save( torch.tensor( batches_size ), 
                    '{}/train_batches_size/train_batches_size_epoch_{}.pt'.format(
                                                                                    report_path,
                                                                                    epoch
                                                                                 )
                  )
        torch.save( torch.tensor( losses ), 
                    '{}/train_losses/train_losses_epoch_{}.pt'.format(
                                                                        report_path,
                                                                        epoch
                                                                     )
                  )
        torch.save( torch.tensor( accuracies ), 
                    '{}/train_accuracies/train_accuracies_epoch_{}.pt'.format(
                                                                                report_path,
                                                                                epoch
                                                                             )
                  )
        if saving_model_every_epoch:
            resnet_save( 
                        '{}/models'.format( report_path ),
                        'vqanet_epoch_{}'.format( epoch ),
                        resnet,
                        optimizer=optimizer
                       )

def resnet_eval( 
                resnet,
                eval_data,
                criterion,
                report_path,
                epoch,
                device,
                batch_size=2,
                num_workers=2,
                check_counter=4,
                gpu=False,
                eval_mode='test'):

    assert eval_mode in ['val', 'test'], 'eval mode must be \'val\' or \'test\''

    utility.mkdir(report_path, '{}_batches_size'.format(eval_mode))
    utility.mkdir(report_path, '{}_losses'.format(eval_mode))
    utility.mkdir(report_path, '{}_accuracies'.format(eval_mode))

    data_loader = DataLoader(   eval_data,
                                batch_size=batch_size,
                                shuffle=False,
                                pin_memory=gpu and torch.cuda.is_available(),
                                num_workers=num_workers)
    resnet = resnet.eval()

    batches_size = list()
    losses = list()
    accuracies = list()

    curr_loss = 0
    for ix,(X, Y) in enumerate( data_loader ):
        X, Y = V(X), V(Y)
        
        if gpu:
            if device=='multi':
                X, Y = nn.DataParallel(X), nn.DataParallel(Y)
            else:
                X, Y = X.cuda(device=device), Y.cuda(device=device)
        
        output = resnet( (X,None) )
        
        if isinstance( criterion, nn.CrossEntropyLoss ):
            Y = torch.argmax( Y, dim=1 )

        loss = criterion(output, Y)
        acc = resnet_accuracy(output, Y)
        

        prev_loss = curr_loss
        curr_loss = loss.item()

        batches_size.append( output.size()[0] )
        losses.append( curr_loss )
        accuracies.append( acc )


        print( 'batch=%d(x%d), prev_loss=%.3f, curr_loss=%.3f, delta=%.3f, acc=%.3f%%' % (
                                                                                            ix,
                                                                                            output.size()[0],
                                                                                            prev_loss,
                                                                                            curr_loss,
                                                                                            curr_loss-prev_loss,
                                                                                            acc*100
                                                                                          ) )
        if ix%check_counter==(check_counter-1):
            # print()
            pass
            
        del X, Y, output
        torch.cuda.empty_cache()
        gc.collect()

        torch.save( torch.tensor( batches_size ), 
                    '{}/{}_batches_size/{}_batches_size_epoch_{}.pt'.format(
                                                                            report_path,
                                                                            eval_mode,
                                                                            eval_mode,
                                                                            epoch
                                                                           )
                  )
        torch.save( torch.tensor( losses ), 
                    '{}/{}_losses/{}_losses_epoch_{}.pt'.format(
                                                                report_path,
                                                                eval_mode,
                                                                eval_mode,
                                                                epoch
                                                               )
                  )
        torch.save( torch.tensor( accuracies ), 
                    '{}/{}_accuracies/{}_accuracies_epoch_{}.pt'.format(
                                                                        report_path,
                                                                        eval_mode,
                                                                        eval_mode,
                                                                        epoch
                                                                       )
                  )

