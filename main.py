import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import data
import model

from utils import batchify, get_batch, repackage_hidden
from adv_drop_algorithms import adversarial_dropout

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/', help='location of the data corpus')
parser.add_argument('--method', type=str, default='AdD', help='FD|AdD')
parser.add_argument('--model', type=str, default='LSTM', help='type of recurrent net (LSTM, QRNN)')
parser.add_argument('--emsize', type=int, default=400, help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150, help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3, help='number of layers')

parser.add_argument('--lr', type=float, default=30, help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--epochs', type=int, default=750, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N', help='batch size')
parser.add_argument('--bptt', type=int, default=70, help='sequence length')

parser.add_argument('--adv_target', type=str, default='w', help='adv dropout target (e, i, h, o, w)')
parser.add_argument('--adv_target_layer', type=int, default=3, help='adv dropout target layer')
parser.add_argument('--adv_change_rates', type=float, default=0.06, help='adv dropout change rates')

parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.4,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--vdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')

parser.add_argument('--rampup_len', type=float, default=100, 
                    help='kappa penalty for hidden states discrepancy (kappa = 0 means no penalty)')
parser.add_argument('--rampup_start', type=float, default=0, 
                    help='kappa penalty for hidden states discrepancy (kappa = 0 means no penalty)')

parser.add_argument('--tied', action='store_false',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=321,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=60,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save_dir', type=str,  default='output/',
                    help='dir path to save the log and the final model')

randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--name', type=str,  default=randomhash, help='path to save the final model')
parser.add_argument('--alpha', type=float, default=0, help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=0, help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--kappa', type=float, default=0.1, help='kappa constant for the regularization (kappa = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6, help='weight decay applied to all weights')
args = parser.parse_args()

args.model_file_name = '/' + args.name + '.pt'
args.log_file_name = '/' + args.name + '.log'

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

eval_batch_size = 10
test_batch_size = 1
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)
if args.cuda:
    model.cuda()
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in model.parameters())
print('Args:', args)
print('Model total parameters:', total_params)

import os
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

string_args = ''
for name in sorted(vars(args)):
    string_args += name + '=' + str(getattr(args, name)) + ', '
string_args += 'total_params=' + str(total_params)
    
with open(args.save_dir + args.log_file_name, 'a') as f:
    f.write(string_args + '\n')
    f.write('epoch time training_running_ppl validation_pll\n')

criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################
def rampup(epoch, start_epoch, rampup_length):
    if epoch < start_epoch:
        return 0
    elif (epoch - start_epoch) < rampup_length:
        p = max(0.0, float(epoch-start_epoch)) / float(rampup_length)
        p = 1.0 - p
        return math.exp(-p*p*5.0)
    else:
        return 1.0

def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)

def generate_masks(data, cur_kappa, is_second=False):
    
    temp_tensor = torch.ones(1, 1).cuda()
    maskw = [ Variable( temp_tensor.new().resize_( (4*args.nhid, args.nhid) ).bernoulli_(1-args.wdrop)/(1-args.wdrop)).cuda()
                 for l in range(0, args.nlayers-1)]
    maskw.append( Variable( temp_tensor.new().resize_( (4*args.emsize, args.emsize) ).bernoulli_(1-args.wdrop)/(1-args.wdrop)).cuda() )
    
    if args.method != 'AdD' or cur_kappa == 0:
        return [None, None, None, maskw, None], None, 0
    else:
        maske = Variable( temp_tensor.new().resize_( (ntokens, 1) ).bernoulli_(1-args.dropoute)/(1-args.dropoute)).cuda()
        maski = Variable( temp_tensor.new().resize_( (1, data.size(1), args.emsize) ).bernoulli_(1-args.dropouti)/(1-args.dropouti)).cuda()
        maskh = [ Variable( temp_tensor.new().resize_( (1, data.size(1), args.nhid) ).bernoulli_(1-args.dropouth)/(1-args.dropouth)).cuda()
                 for l in range(0, args.nlayers-1)] 
        masko = Variable( temp_tensor.new().resize_( (1, data.size(1), args.emsize) ).bernoulli_(1-args.dropout)/(1-args.dropout)).cuda()
        
        if args.adv_target == 'e':
            maske = Variable( torch.ones(ntokens, 1).cuda() , requires_grad=is_second).cuda()
            target = maske
            target_size = ntokens
        elif args.adv_target == 'i':
            maski = Variable(torch.ones(1, data.size(1), args.emsize).cuda(), requires_grad=is_second)
            target = maski
            target_size = args.emsizec
        elif args.adv_target == 'h':
            maskh[args.adv_target_layer-1] = Variable(torch.ones(1, data.size(1), args.emsize).cuda(), requires_grad=is_second)
            target = maskh[args.adv_target_layer-1] 
            target_size = args.emsize
        elif args.adv_target == 'w':
            size = args.emsize if args.adv_target_layer == args.nlayers else args.nhid
            maskw[args.adv_target_layer-1] = Variable(torch.ones(4*size, size).cuda(), requires_grad=is_second)
            target = maskw[args.adv_target_layer-1]
            target_size = size
        elif args.adv_target == 'o':
            masko = Variable(torch.ones(1, data.size(1), args.emsize).cuda(), requires_grad=is_second)
            target = masko
            target_size = args.emsize
        return [maske, maski, maskh, maskw, masko], target, target_size

def assign_masks(masks, target):
    if args.adv_target == 'e':
        masks[0] = Variable(target).cuda()
    elif args.adv_target == 'i':
        masks[1] = Variable(target).cuda()
    elif args.adv_target == 'h':
        masks[2][args.adv_target_layer-1] = Variable(target).cuda()
    elif args.adv_target == 'w':
        masks[3][args.adv_target_layer-1] = Variable(target).cuda()
    elif args.adv_target == 'o':
        masks[4] = Variable(target).cuda()
    return masks
            
def train(epoch):
    # Turn on training mode which enables dropout.
    if args.model == 'QRNN': model.reset()
    total_loss, total_k_loss = 0, 0
    train_running_loss, train_k_running_loss = 0, 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    batch, i = 0, 0
    cur_kappa = args.kappa * rampup(epoch, args.rampup_start, args.rampup_len)
    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + 10)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        # generate masks
        [maske, maski, maskh, maskw, masko], _, _ = generate_masks(data, cur_kappa,  False)

        output, new_hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True, maske=maske, maski=maski, maskh=maskh, maskw=maskw, masko=masko)
        raw_loss = criterion(output.view(-1, ntokens), targets)

        total_loss += raw_loss.data
        train_running_loss += raw_loss.data
        
        loss = raw_loss
        
        # Activiation Regularization
        loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        
        # Kappa penalty 
        if cur_kappa > 0:
            
            # initial dropout mask
            masks, target_m, target_m_size = generate_masks(data, cur_kappa, True)
            
            if args.method == "AdD":
                kappa_output, _, _, _ = model(data, hidden, return_h=True, maske=masks[0], maski=masks[1], maskh=masks[2], maskw=masks[3], masko=masks[4])
        
                l2_kappa = (output.detach() - kappa_output).pow(2).mean()
                raw_k_loss = criterion(kappa_output.view(-1, ntokens), targets)
                additional_loss = l2_kappa # raw_k_loss/2 + args.kappa * 
                
                additional_loss.backward()
                Jacob = target_m.grad.data.clone()
                target_m = target_m.data.clone()
                target_m = adversarial_dropout(target_m, Jacob, target_m_size, int( target_m_size*args.adv_change_rates ) + 1)
                masks = assign_masks(masks, target_m)
                optimizer.zero_grad()
            
            kappa_output, _, _, _ = model(data, hidden, return_h=True, maske=masks[0], maski=masks[1], maskh=masks[2], maskw=masks[3], masko=masks[4])
            raw_k_loss = criterion(kappa_output.view(-1, ntokens), targets)
            l2_kappa = (output.detach() - kappa_output).pow(2).mean()
            
            total_k_loss += raw_k_loss.data
            train_k_running_loss += raw_k_loss.data
            
            additional_loss = cur_kappa * l2_kappa 
            loss = raw_loss + additional_loss
        else:
            total_k_loss += raw_loss.data*args.kappa
            train_k_running_loss += raw_loss.data*args.kappa
        hidden = new_hidden
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()

        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            cur_k_loss = total_k_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | f_ppl {:8.2f} | s_ppl {:8.2f} '.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), math.exp(cur_k_loss)))
            total_loss = 0
            total_k_loss = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len
    return train_running_loss[0] / batch, train_k_running_loss[0] / batch

# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000

# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train_running_loss, train_k_running_loss = train(epoch)
        if 't0' in optimizer.param_groups[0]:
            tmp = {}
            for prm in model.parameters():
                tmp[prm] = prm.data.clone()
                prm.data = optimizer.state[prm]['ax'].clone()

            val_loss2 = evaluate(val_data)
            val = val_loss2
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss2, math.exp(val_loss2)))
            print('-' * 89)

            if val_loss2 < stored_loss:
                with open(args.save_dir + args.model_file_name, 'wb') as f:
                    torch.save(model, f)
                print('Saving Averaged!')
                stored_loss = val_loss2

            for prm in model.parameters():
                prm.data = tmp[prm].clone()

        else:
            val_loss = evaluate(val_data, eval_batch_size)
            val = val_loss
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss, math.exp(val_loss)))
            print('-' * 89)

            if val_loss < stored_loss:
                with open(args.save_dir + args.model_file_name, 'wb') as f:
                    torch.save(model, f)
                print('Saving Normal!')
                stored_loss = val_loss

            if 't0' not in optimizer.param_groups[0] and (len(best_val_loss)>args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                print('Switching!')
                optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
                #optimizer.param_groups[0]['lr'] /= 2.
            best_val_loss.append(val_loss)
        with open(args.save_dir + args.log_file_name, 'a') as f:
            f.write(str(epoch) + ' ' + str(time.time() - epoch_start_time) + ' ' + 
                    str(math.exp(train_running_loss)) +' '+ str(math.exp(train_k_running_loss)) +  ' ' + str(math.exp(val)) + '\n')

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save_dir + args.model_file_name, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss = evaluate(test_data, test_batch_size)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

with open(args.save_dir + args.log_file_name, 'a') as f:
    f.write(str(0) + ' ' + str(0) + ' ' + str(math.exp(test_loss)) + ' ' + str(math.exp(test_loss)) + '\n')
