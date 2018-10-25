# adapted from Magda Paschali's Github
# https://github.com/MaggiePas/ManiFool/blob/master/Dermofit_train.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from tensorboardX import SummaryWriter
from utils import *
from median_frequency_balancing import *

# Parse arguments if given
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument(
	'--epochs', '-epoch', default=50, type=int, help='number of epochs')
parser.add_argument(
	'--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
num_epochs = 50
model_path = './checkpoint/ckpt_test_rs18_sgd_da_001.t7'

# Prepare and load our dataset
print('==> Preparing data..')
transform_train = transforms.Compose([
	transforms.Resize((224, 224)),
	transforms.RandomHorizontalFlip(),
	transforms.RandomRotation(degrees=60),
	# transforms.ColorJitter(),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
	transforms.Resize((224, 224)),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

adversarials = np.load("adversarial.npy")
adversarials_labels = np.load("adversarial_labels.npy")

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

"""trainset = torchvision.datasets.ImageFolder(
	root=
	'/home/magda/Documents/PyTorch/ManiFool/data/Dermofit/dermofit_train_new/',
	transform=transform_train)
trainloader = torch.utils.data.DataLoader(
	trainset, batch_size=64, shuffle=True, num_workers=2)

testset = torchvision.datasets.ImageFolder(
	root='/home/magda/Documents/PyTorch/ManiFool/data/Dermofit/dermofit_test/',
	transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

classes = ('Actinic_Keratosis', 'Basal_Cell_Carcinoma', 'Dermatofibroma',
		   'Haemangioma', 'Intraepithelial_Carcinoma', 'Malignant_Melanoma',
		   'Melanocytic_Nevus', 'Pyogenic_Granuloma', 'Seborrhoeic_Keratosis',
		   'Squamous_Cell_Carcinoma')"""

"""# Run this piece of code if you want to calculate the weights of each class using Median Frequency Balancing
all_targets = []
count_targets = 0
#all_targets = np.zeros((584, 1))
all_targets = np.zeros((653, 1))

for batch_idx, (inputs, targets) in enumerate(trainloader):
		inputs, targets = inputs.to(device), targets.to(device)
		all_targets[batch_idx, 0] = targets
		count_targets = count_targets + len(targets)

weights = median_frequency_balancing(all_targets)
print('Weights: ', weights)"""

# Build the model
print('==> Building model..')

net = torchvision.models.resnet18(pretrained=True)
## Change the last layer since we don't have 1000 classes but we want to used a pretrained model
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, len(classes))

net = net.to(device)
if device == 'cuda':
	net = torch.nn.DataParallel(net)
	cudnn.benchmark = True

# Resume training from checkpoint
if args.resume:
	# Load checkpoint.
	print('==> Resuming from checkpoint..')
	assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
	checkpoint = torch.load('./checkpoint/ckpt_dermofit.t7')
	net.load_state_dict(checkpoint['net'])
	best_acc = checkpoint['acc']
	start_epoch = checkpoint['epoch']

# Training Parameters and Optimization
# Weights for train - test - validation split
# weights  = np.asarray([1.85, 0.34259259, 1.27586207, 0.84090909, 1.05714286, 1.08823529, 0.24832215, 3.7, 0.31896552, 0.94871795])

# Tensorboard Writer
writer = SummaryWriter()

# Weights for train - test split
"""weights = np.asarray([
	1.80434783, 0.34583333, 1.25757576, 0.84693878, 1.06410256, 1.09210526,
	0.25, 3.45833333, 0.32170543, 0.94318182
])

weights = torch.from_numpy(weights).type(torch.FloatTensor).to(
	torch.device(device))"""

criterion = nn.CrossEntropyLoss()  # weight=weights)
optimizer = optim.SGD(
	net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
# optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)

writer.add_text('Optimizer', 'SGD', 0)
writer.add_text('Momentum', '0.9', 0)
writer.add_text('Weight_decay', '5e-4', 0)

# Decaying learning rate
scheduler = optim.lr_scheduler.MultiStepLR(
	optimizer, milestones=[20, 40, 50], gamma=0.1)


# Training
def train(epoch, writer, perc_adversarial=0.1):
	print('\nEpoch: %d' % epoch)
	scheduler.step()
	net.train()
	train_loss = 0
	correct = 0
	total = 0

	for param_group in optimizer.param_groups:
		# print(param_group['lr'])
		writer.add_scalars('data/params', {'learning_rate': param_group['lr']}, epoch)

	count = 0
	for batch_idx, (inputs, targets) in enumerate(trainloader):
		inputs, targets = inputs.to(device), targets.to(device)
		optimizer.zero_grad()
		outputs = net(inputs)
		loss = criterion(outputs, targets)
		loss.backward()
		optimizer.step()

		train_loss += loss.item()
		_, predicted = outputs.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().item()

		progress_bar(batch_idx, len(trainloader), '\nLoss: %.3f | Acc: %.3f%% (%d/%d)' %
					(train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

		if epoch > 1:
			writer.add_scalars(
				'data/loss', {'Train_Loss': loss.item()},
				batch_idx * trainloader.batch_size + (
						(epoch - 1) * len(trainloader) * trainloader.batch_size))
			writer.add_scalars(
				'data/accuracy', {'Train_Accuracy': correct / total},
				batch_idx * trainloader.batch_size + (
						(epoch - 1) * len(trainloader) * trainloader.batch_size))
		else:
			writer.add_scalars('data/loss', {'Train_Loss': loss.item()}, batch_idx * trainloader.batch_size)
			writer.add_scalars('data/accuracy', {'Train_Accuracy': correct / total}, batch_idx * trainloader.batch_size)

		count += 1
		if count % perc_adversarial * 100 == 0:
			# every perc_adversarial*100 iterations, one adds a batch of adversarial examples into the training
			count = 0
			idxs = np.random.choice(np.arange(np.shape(adversarials_labels)[0]), size=(trainloader.batch_size))
			input_adv = torch.from_numpy(adversarials[idxs])
			targets_adv = torch.from_numpy(adversarials_labels[idxs])
			optimizer.zero_grad()
			outputs_adv = net(input_adv)
			loss_adv = criterion(outputs_adv, targets_adv)
			loss_adv.backward()
			optimizer.step()
			print("trained with a batch of adversarial examples")


def test(epoch, writer):
	global best_acc
	net.eval()
	test_loss = 0
	correct = 0
	total = 0
	mean_acc = 0
	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(testloader):
			inputs, targets = inputs.to(device), targets.to(device)
			outputs = net(inputs)
			loss = criterion(outputs, targets)

			test_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()

			mean_acc = mean_acc + correct / total

			progress_bar(
				batch_idx, len(testloader),
				'\nLoss: %.3f | Acc: %.3f%% (%d/%d)' %
				(test_loss / (batch_idx + 1), 100. * correct / total, correct,
				 total))

			if epoch > 1:
				writer.add_scalars(
					'data/loss', {'Test_Loss': loss.item()},
					batch_idx * testloader.batch_size + (
							(epoch - 1) * len(testloader) * testloader.batch_size))
				writer.add_scalars(
					'data/accuracy', {'Test_Accuracy': correct / total},
					batch_idx * testloader.batch_size + (
							(epoch - 1) * len(testloader) * testloader.batch_size))
			else:
				writer.add_scalars('data/loss', {'Test_Loss': loss.item()},
				                   batch_idx * testloader.batch_size)
				writer.add_scalars('data/accuracy',
				                   {'Test_Accuracy': correct / total},
				                   batch_idx * testloader.batch_size)

	# Save checkpoint.
	# acc = 100.*correct/total
	mean_acc = 100. * mean_acc
	if mean_acc > best_acc:
		print('Saving..')
		state = {
			'net': net.state_dict(),
			'acc': mean_acc,
			'epoch': epoch,
		}
		if not os.path.isdir('checkpoint'):
			os.mkdir('checkpoint')
		torch.save(state, model_path)
		best_acc = mean_acc


for epoch in range(start_epoch, num_epochs):
	train(epoch, writer, 0.1)
	test(epoch, writer)

# test_all(model_path)

writer.export_scalars_to_json(model_path[:-2] + 'json')
writer.close()
