from tqdm import tqdm
from losses import AverageMeter, MultiTaskLoss
from metrics import calculate_metrics
from model import  MultiTaskModelTaskSpecificAttn, SingleTaskModel
from torch import optim
import torch
from torch import nn
from dataset import get_dataloaders
import os
import warnings
warnings.filterwarnings("ignore")
import argparse
from torch.optim.lr_scheduler import StepLR


def args_parser():
    parser = argparse.ArgumentParser(description='PyTorch Training')  
    # provide the path of the feature file
    parser.add_argument('--data-path', default="../npy_files_new/Xy_marlin.npy", type=str, help="dataset path")
    parser.add_argument('--train-fold', default="../../train.txt", type=str, help="train fold path")
    parser.add_argument('--test-fold', default="../../test.txt", type=str, help="test fold path")
    parser.add_argument('--epochs', default=500, type=int, help="Number of epochs")
    parser.add_argument('--lr', default=0.000001, type=float, help="Learning rate")
    parser.add_argument('--model-type', default=1, type=int, help="Model type")
    parser.add_argument('--output-dir', default='weights', help='output weights dir')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--task', type=int, default=0,  help="0 for wander, 1 for engage")
    parser.add_argument('--multi-task', action='store_true', help="if training is multi-task")
    
    args = parser.parse_args()
    return args

def train_multitask(model, train_loader, optW, optE, criterionW, criterionE):
    losses = AverageMeter()
    model.train()
    for batch_idx, data in enumerate(train_loader):
        inputs = data['features']
        labels = data['labels']
        
        inputs = inputs.cuda()
        labels = labels.cuda()
    
        outputs = model(inputs.float())
        
        outW, outE = outputs
        lossW = criterionW(outW, labels[:, 0].long()) 
        lossE = criterionE(outE, labels[:, 1].long()) * 6

        optW.zero_grad()
        lossW.backward(retain_graph=True)
        
        optE.zero_grad()
        lossE.backward()
        optW.step()
        optE.step()

        losses.update(lossW.item() + lossE.item(), inputs.size(0)) 
    return losses.avg

def train_singletask(model, train_loader, opt, criterion, task_idx):
    losses = AverageMeter()
    model.train()
    for batch_idx, data in enumerate(train_loader):
        inputs = data['features']
        labels = data['labels']
        
        # to cuda
        inputs = inputs.cuda()
        labels = labels.cuda()

        out = model(inputs.float())
        
        loss = criterion(out, labels[:, task_idx].long()) 
       
        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.update(loss.item() , inputs.size(0))
    return losses.avg

def save_model(args, model, mets, epoch, prefix):
    print ("saving model...")
    torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'mets': mets,
            }, args.output_dir + f'/{prefix}_best.pt')

def main():
    args = args_parser()
    train_loader, test_loader, num_features, class_weights = get_dataloaders(args.data_path, args.train_fold, args.test_fold)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.multi_task:
        print ("Multi task training...")
        if args.model_type == 1:
            model = MultiTaskModelTaskSpecificAttn(num_features)
        else:
            raise ValueError('not available')
    else:
        print ("Single task training...")
        if args.task == 0:
            model = SingleTaskModel(num_features, n_classes=2)
        elif args.task == 1:
            model = SingleTaskModel(num_features, n_classes=4)
        else:
            raise ValueError('invalid task')
        
    
    best_wander_auc = 0
    best_overall = 0

    best_state = {
        'wander': None,
        'both': None
    }

    last_epoch = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        state_dict = checkpoint['state_dict']
        last_epoch = checkpoint['epoch']
        best_wander = checkpoint['mets']['weighted_precision']['wander'] + checkpoint['mets']['weighted_f1']['wander'] + checkpoint['mets']['weighted_recall']['wander']
        best_overall = best_wander + checkpoint['mets']['weighted_precision']['engage'] + checkpoint['mets']['weighted_f1']['engage'] + checkpoint['mets']['weighted_recall']['engage']

        model.load_state_dict(state_dict, strict=True)
        print ("==== Model loaded ! ==== ")

    model = model.cuda()
    if args.multi_task:
        criterionW = nn.CrossEntropyLoss(weight=class_weights[0].cuda())
        criterionE = nn.CrossEntropyLoss(weight=class_weights[1].cuda())

        optW = optim.AdamW(model.parameters(), lr=args.lr)
        optE = optim.AdamW(model.parameters(), lr=args.lr)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights[args.task].cuda())
        opt = optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        _flag = False
        if args.multi_task:
            training_loss = train_multitask(model, train_loader, optW, optE, criterionW, criterionE)
        else:
            training_loss = train_singletask(model, train_loader, opt, criterion, args.task)

        print ("\nEpoch: {}/{}   Loss -  {:.5f}".format(epoch, args.epochs, training_loss))

        if (epoch + 1) % 1 == 0:
            mets = calculate_metrics(model, test_loader, task=args.task)
            if args.multi_task:
                print ("[ENGAGE] accuracy ==> ", mets['acc']['engage'])
                print ("[WANDER] accuracy ==> ", mets['acc']['wander'])
                print ("[ENGAGE] f1 ==> ", mets['f1']['engage'])
                print ("[WANDER] f1 ==> ", mets['f1']['wander'])
                print ("[ENGAGE] roc-auc ==>", mets['auc']['engage'])
                print ("[WANDER] roc-auc ==>", mets['auc']['wander'])
                print ("[ENGAGE] weighted precision ==>", mets['weighted_precision']['engage'])
                print ("[WANDER] weighted precision ==>", mets['weighted_precision']['wander'])
                print ("[ENGAGE] weighted f1 ==>", mets['weighted_f1']['engage'])
                print ("[WANDER] weighted f1 ==>", mets['weighted_f1']['wander'])
                print ("[ENGAGE] weighted recall ==>", mets['weighted_recall']['engage'])
                print ("[WANDER] weighted recall ==>", mets['weighted_recall']['wander'])
                overall = mets['weighted_precision']['wander'] + mets['weighted_f1']['wander'] + mets['weighted_recall']['wander'] + mets['weighted_precision']['engage'] + mets['weighted_f1']['engage'] + mets['weighted_recall']['engage'] 
                
                if  overall > best_overall:
                    print("overall:", overall)
                    print("best_overall:", best_overall)
                    best_overall = overall
                    
                    print ("[ENGAGE] weighted precision ==>", mets['weighted_precision']['engage'])
                    print ("[WANDER] weighted precision ==>", mets['weighted_precision']['wander'])
                    print ("[ENGAGE] weighted f1 ==>", mets['weighted_f1']['engage'])
                    print ("[WANDER] weighted f1 ==>", mets['weighted_f1']['wander'])
                    print ("[ENGAGE] weighted recall ==>", mets['weighted_recall']['engage'])
                    print ("[WANDER] weighted recall ==>", mets['weighted_recall']['wander'])

                    print ("\n<== [ENGAGE] classification report ==>")
                    print (mets['report']['engage'])

                    print ("<== [WANDER] classification report ==>")
                    print (mets['report']['wander'])

                    best_state['both'] = mets
                    save_model(args, model, mets, last_epoch+epoch, 'best')
            else:
                print("weighted_precision + weighted_f1 + weighted_recall ===>", mets['weighted_precision']+mets['weighted_f1']+mets['weighted_recall'])
                overall = mets['weighted_precision'] + mets['weighted_f1'] + mets['weighted_recall']
                print ("Last best weighted_precision + weighted_f1 + weighted_recall: ", best_overall)

                if overall > best_overall:
                    best_overall = overall
                    print ("weighted_precision ==>", mets['weighted_precision'])
                    print ("weighted_f1 ==>", mets['weighted_f1'])
                    print ("weighted_recall ==>", mets['weighted_recall'])
                    print ("\n<== classification report ==>")
                    print (mets['report'])
                    best_state['both'] = mets
                    save_model(args, model, mets, last_epoch+epoch, 'best')

    print ("<==== Final results [model={}] [fold={}]====>\n".format(args.model_type, args.train_fold))
    mets = best_state['both']

    if args.multi_task:
        print ("[ENGAGE] accuracy ==> ", mets['acc']['engage'])
        print ("[WANDER] accuracy ==> ", mets['acc']['wander'])
        print ("[ENGAGE] f1 ==> ", mets['f1']['engage'])
        print ("[WANDER] f1 ==> ", mets['f1']['wander'])
        print ("[ENGAGE] roc-auc ==>", mets['auc']['engage'])
        print ("[WANDER] roc-auc ==>", mets['auc']['wander'])
        print ("[ENGAGE] weighted precision ==>", mets['weighted_precision']['engage'])
        print ("[WANDER] weighted precision ==>", mets['weighted_precision']['wander'])
        print ("[ENGAGE] weighted f1 ==>", mets['weighted_f1']['engage'])
        print ("[WANDER] weighted f1 ==>", mets['weighted_f1']['wander'])
        print ("[ENGAGE] weighted recall ==>", mets['weighted_recall']['engage'])
        print ("[WANDER] weighted recall ==>", mets['weighted_recall']['wander'])
        print ("[ENGAGE] class-wise accuracy ==> ", mets['class_acc']['engage']) 
        print ("[WANDER] class-wise accuracy ==> ", mets['class_acc']['wander']) 
        print ("\n<== [ENGAGE] classification report ==>")
        print (mets['report']['engage'])

        print ("\n<== [WANDER] classification report ==>")
        print (mets['report']['wander'])
    else:
        print ("accuracy ===>", mets['acc'])
        print ("f1 ===>", mets['f1'])
        print ("roc-auc ===>", mets['auc'])
        print ("weighted precision ==>", mets['weighted_precision'])
        print ("weighted f1 ==>", mets['weighted_f1'])
        print ("weighted recall ==>", mets['weighted_recall'])
        print ("class-wise accuracy ===>", mets['class_acc'])
        print ("\n<== classification report ==>")
        print (mets['report'])

if __name__ == '__main__':
    main()
