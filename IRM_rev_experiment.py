# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 12:21:27 2022

@author: dohyeon
"""

import argparse
import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim, autograd
import ipdb

parser = argparse.ArgumentParser(description='Colored MNIST')
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--l2_regularizer_weight', type=float,default=0.001)
parser.add_argument('--lr', type=float, default=0.0075)
parser.add_argument('--n_restarts', type=int, default=20)
parser.add_argument('--penalty_anneal_iters', type=int, default=100)
parser.add_argument('--penalty_weight', type=float, default=10000.0)
parser.add_argument('--steps', type=int, default=501)
parser.add_argument('--grayscale_model', action='store_true')
flags = parser.parse_args()

def cal_params_binary(uni_env, extra_mdls):
    idx_0 = torch.where(uni_env['labels']==0)[0]
    idx_1 = torch.where(uni_env['labels']==1)[0]

    num_ks = (idx_0.shape[0], idx_1.shape[0])


    nll_0s = []
    nll_1s = []
    #ipdb.set_trace()
    for uni_mdl in extra_mdls:
        uni_mdl.eval()
        nll_0s.append(mean_log_cost2(uni_mdl(uni_env['images'][idx_0]), uni_env['labels'][idx_0]))
        nll_1s.append(mean_log_cost2(uni_mdl(uni_env['images'][idx_1]), uni_env['labels'][idx_1]))


    rek = torch.tensor([idx_0.shape[0],idx_1.shape[0]])/uni_env['labels'].shape[0]
    #ipdb.set_trace()
    sek = torch.tensor([torch.mean(torch.tensor(nll_0s)), torch.mean(torch.tensor(nll_1s))])
    return rek, sek

def get_der(uni_env, uni_mdl, pkn, lamda1):

    idx_0 = torch.where(uni_env['labels']==0)[0]
    idx_1 = torch.where(uni_env['labels']==1)[0]

    output1 = ((uni_mdl(uni_env['images']+0.00001)-uni_mdl(uni_env['images']))/0.00001)**2

    #ipdb.set_trace()
    return (output1[idx_0]*pkn[0]*lamda1).sum()+(output1[idx_1]*pkn[1]*lamda1).sum()




print('Flags:')
for k,v in sorted(vars(flags).items()):
  print("\t{}: {}".format(k, v))

final_train_accs = []
final_test_accs = []
final_train_losses = []
final_test_losses = []
final_pk1s = []
final_lk1s = []
extra_models = []

pk1, lk1 = 1, 1
for restart in range(flags.n_restarts):
    print("Restart", restart)
    
    # Load MNIST, make train/val splits, and shuffle train set examples
    
    mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
    mnist_train = (mnist.data[:50000], mnist.targets[:50000])
    mnist_val = (mnist.data[50000:], mnist.targets[50000:])
    
    rng_state = np.random.get_state()
    np.random.shuffle(mnist_train[0].numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(mnist_train[1].numpy())
    
    # Build environments

    def make_environment(images, labels, e):
      def torch_bernoulli(p, size):
        return (torch.rand(size) < p).float()
      def torch_xor(a, b):
        return (a-b).abs() # Assumes both inputs are either 0 or 1
      # 2x subsample for computational convenience
      images = images.reshape((-1, 28, 28))[:, ::2, ::2]
      # Assign a binary label based on the digit; flip label with probability 0.25
      labels = (labels < 5).float()
      # Assign a color based on the label; flip the color with probability e
      colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
      # Apply the color to the image by zeroing out the other color channel
      images = torch.stack([images, images], dim=1)
      images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0
      #ipdb.set_trace()

      return {
        'images': (images.float() / 255.).cuda(),
        'labels': labels[:, None].cuda()
      }

    def make_environment1(images, labels, e):
      def torch_bernoulli(p, size):
        return (torch.rand(size) < p).float()
      def torch_xor(a, b):
        return (a-b).abs() # Assumes both inputs are either 0 or 1
      # 2x subsample for computational convenience
      images = images.reshape((-1, 28, 28))[:, ::2, ::2]
      # Assign a binary label based on the digit; flip label with probability 0.25
      labels = (labels < 5).float()
      labels = torch_xor(labels, torch_bernoulli(0.5, len(labels)))
      # Assign a color based on the label; flip the color with probability e
      colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
      # Apply the color to the image by zeroing out the other color channel
      images = torch.stack([images, images], dim=1)
      images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0
      #ipdb.set_trace()

      return {
        'images': (images.float() / 255.).cuda(),
        'labels': labels[:, None].cuda()
      }


    envs = [
      make_environment(mnist_train[0][::2], mnist_train[1][::2], 0.1),
      make_environment1(mnist_train[0][1::2], mnist_train[1][1::2], 0.2),
      make_environment(mnist_val[0], mnist_val[1], 0.9)
    ]


    if restart> 0:
        rek1, sek1 = cal_params_binary(envs[0], extra_models)
        rek2, sek2 = cal_params_binary(envs[1], extra_models)

        te1_const1 = envs[0]['labels'].shape[0]/(4**(7/5))
        te1_val1 = torch.tensor([(rek1[0]/sek1[0])**4/5 +  (rek1[1]/sek1[1])**4/5])**-1

        te2_const1 = envs[1]['labels'].shape[0]/(4**(7/5))
        te2_val1 = torch.tensor([(rek2[0]/sek2[0])**4/5 +  (rek2[1]/sek2[1])**4/5])**-1


        lam1 = ((1/envs[0]['labels'].shape[0]) + (1/envs[1]['labels'].shape[0]))**(2/5)



        tt1 = te1_const1 * te1_val1
        tt2 = te2_const1 * te2_val1


        final_lk1s.append(torch.tensor([tt1, tt2]))



        tt3=(1/(4**(7/5))) * (
            (sek1[0]/rek1[0])**(4/5) + (sek2[0]/rek2[0])**(4/5))

        tt4=(1/(4**(7/5))) * (
            (sek1[1]/rek1[1])**(4/5) + (sek2[1]/rek2[1])**(4/5))

        final_pk1s.append(torch.tensor([tt3,tt4]))

    else:
        tt1=1
        tt2=1
        tt3=1
        tt4=1
        lam1 = ((1/envs[0]['labels'].shape[0]) + (1/envs[1]['labels'].shape[0]))**(2/5)


    # Define and instantiate the model

    class MLP(nn.Module):
      def __init__(self):
        super(MLP, self).__init__()
        if flags.grayscale_model:
          lin1 = nn.Linear(14 * 14, flags.hidden_dim)
        else:
          lin1 = nn.Linear(2 * 14 * 14, flags.hidden_dim)
        lin2 = nn.Linear(flags.hidden_dim, flags.hidden_dim)
        lin3 = nn.Linear(flags.hidden_dim, 1)
        for lin in [lin1, lin2, lin3]:
          nn.init.xavier_uniform_(lin.weight)
          nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)
      def forward(self, input):
        if flags.grayscale_model:
          out = input.view(input.shape[0], 2, 14 * 14).sum(dim=1)
        else:
          out = input.view(input.shape[0], 2 * 14 * 14)
        out = self._main(out)
        return out
    
    mlp = MLP().cuda()

  # Define loss function helpers

    def mean_nll(logits, y):
      return nn.functional.binary_cross_entropy_with_logits(logits, y)
    

    def mean_log_cost1(logits1, y1):
        #ipdb.set_trace()
        if y1[0] == 1:
            output=-torch.log(logits1).mean()
        else:
            output = -torch.log(1-logits1).mean()
        print(output)

        return output

    def mean_log_cost2(logits2, y2):
        hy = torch.sigmoid(logits2).squeeze()

        #idx_01 = torch.where(hy==0)[0]
        #idx_11 = torch.where(hy==1)[0]

        idx_01 = torch.where(y2==0)[0]
        idx_11 = torch.where(y2==1)[0]


        result_tensor = torch.zeros(y2.shape[0]).cuda()
        #ipdb.set_trace()

        result_tensor[idx_01] = torch.nan_to_num( torch.log(1-hy[idx_01]), posinf=0, neginf=0)
        result_tensor[idx_11] = torch.nan_to_num(torch.log(hy[idx_11]), posinf=0, neginf=0)

        return -result_tensor.mean()


    def mean_accuracy(logits, y):
      preds = (logits > 0.).float()
      return ((preds - y).abs() < 1e-2).float().mean()
    
    def penalty(logits, y):
      scale = torch.tensor(1.).cuda().requires_grad_()
      loss = mean_nll(logits * scale, y)
      grad = autograd.grad(loss, [scale], create_graph=True, retain_graph=True)[0]
      return torch.sum(grad**2)
    
    # Train loop

    def pretty_print(*values):
      col_width = 13
      def format_val(v):
        if not isinstance(v, str):
          v = np.array2string(v, precision=5, floatmode='fixed')
        return v.ljust(col_width)
      str_values = [format_val(v) for v in values]
      print("   ".join(str_values))
    
    optimizer = optim.Adam(mlp.parameters(), lr=flags.lr)
    
    pretty_print('step', 'train nll', 'train acc', 'train penalty', 'test acc')
    
    for step in range(flags.steps):
        for env in envs:
            logits = mlp(env['images'])
            env['nll'] = mean_nll(logits, env['labels'])
            env['acc'] = mean_accuracy(logits, env['labels'])
            env['penalty'] = penalty(logits, env['labels'])
            
        train_nll = torch.stack([envs[0]['nll'], envs[1]['nll']]).mean()
        train_acc = torch.stack([envs[0]['acc'], envs[1]['acc']]).mean()
        #train_penalty = torch.stack([envs[0]['penalty'], envs[1]['penalty']]).mean()

        train_penalty = torch.stack([envs[0]['penalty'], envs[1]['penalty']])

        second_loss = get_der(envs[0], mlp, [tt3,tt4], lam1) + get_der(envs[1], mlp, [tt3,tt4], lam1)

        weight_norm = torch.tensor(0.).cuda()
        for w in mlp.parameters():
          weight_norm += w.norm().pow(2)
          
        loss = train_nll.clone()

        loss += weight_norm * flags.l2_regularizer_weight

        loss += second_loss

        penalty_weight = torch.tensor([tt1, tt2]).cuda()

        loss += (penalty_weight * train_penalty).mean()
        if (penalty_weight.sum()/2) > 1.0:
          # Rescale the entire loss to keep gradients in a reasonable range
          loss /= (penalty_weight.sum()/2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        test_acc = envs[2]['acc']
        test_nll = envs[2]['nll']
        if step % 100 == 0:
            pretty_print(
              np.int32(step),
              train_nll.detach().cpu().numpy(),
              train_acc.detach().cpu().numpy(),
              train_penalty.detach().cpu().numpy(),
              test_acc.detach().cpu().numpy()
            )

    final_train_losses.append(train_nll.detach().cpu().numpy())
    final_test_losses.append(test_nll.detach().cpu().numpy())

    extra_models.append(mlp)
    final_train_accs.append(train_acc.detach().cpu().numpy())
    final_test_accs.append(test_acc.detach().cpu().numpy())
    print('Final train acc (mean/std across restarts so far):')
    print(np.mean(final_train_accs), np.std(final_train_accs))
    print('Final test acc (mean/std across restarts so far):')
    print(np.mean(final_test_accs), np.std(final_test_accs))
    #%%

    import matplotlib.pyplot as plt
    plt.title("LipIRM")
    plt.xlabel("Restart")
    plt.ylabel("Accuracy")
    plt.plot(final_train_accs, label='train')
    plt.plot(final_test_accs, label='test')
    plt.legend(loc='upper right')

    #%%
    plt.plot([xx for xx,yy in final_pk1s], label='class 0'), plt.plot([yy for xx,yy in final_pk1s],label='class 1'),plt.legend()
    plt.xlabel("Restart")
    plt.ylabel("Sample-level penalty weight")


    #%%

    plt.plot([xx for xx,yy in final_lk1s], label='env1'), plt.plot([yy for xx,yy in final_lk1s],label='env2'),plt.legend()
    plt.xlabel("Restart")
    plt.ylabel("Domain-level penalty weight")
