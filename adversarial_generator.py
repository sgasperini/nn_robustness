import foolbox
import torch
import torchvision.models as models
import numpy as np
from utils import *
import torchvision
import torchvision.transforms as transforms


transform_train = transforms.Compose([
	transforms.Resize((224, 224)),
	transforms.RandomHorizontalFlip(),
	transforms.RandomRotation(degrees=60),
	#transforms.ColorJitter(),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
	transforms.Resize((224, 224)),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# instantiate the model
resnet18 = models.resnet18(pretrained=True).eval()
if torch.cuda.is_available():
	resnet18 = resnet18.cuda()
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, len(classes))

mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
std = np.array([0.2023, 0.1994, 0.2010]).reshape((3, 1, 1))
fmodel = foolbox.models.PyTorchModel(
	resnet18, bounds=(-255, 255), num_classes=len(classes), preprocessing=(mean, std))
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ THE BOUNDS...

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# get source image and label

attacks = [foolbox.attacks.FGSM(fmodel), foolbox.attacks.ProjectedGradientDescentAttack(fmodel),
	foolbox.attacks.CarliniWagnerL2Attack(fmodel)]
names = ['FGSM', 'PGD', 'CW']
# apply attack on source image

adversarials = []
adversarials_labels = []

for i in range(0, 1):  # range(len(attacks)):
	# attack = foolbox.attacks.FGSM(fmodel)

	for n in range(0, 50):

		for batch_idx, (inputs, targets) in enumerate(trainloader):
			inputs, targets = inputs.to(device), targets.to(device)

			image = inputs[0].numpy()
			# torch_tensor = torch.from_numpy(np_array)

			label = targets[0].numpy()

			#image, label = foolbox.utils.imagenet_example(data_format='channels_first')
			#image = image / 255.  # because our model expects values in [0, 1]

			# print(np.max(image))

			# import mat plotlib.pyplot as plt
			# plt.imshow(image)

			print('\nlabel', label)
			print('predicted class', np.argmax(fmodel.predictions(image)))

			print('running ' + names[i] + '...')

			adversarial = attacks[i](image, label) if names[i] != 'PGD' \
				else attacks[i](image, label, iterations=40, epsilon=0.3)

			adversarials.append(adversarial)
			adversarials_labels.append(label)

			# print(np.shape(adversarial))

			print(names[i] + ' adversarial class', np.argmax(fmodel.predictions(adversarial)))


adversarials = np.asarray(adversarials)
adversarials_labels = np.asarray(adversarials_labels)

np.save('adversarials.npy', adversarials)
np.save('adversarials_labels.npy', adversarials_labels)

print("\ndone")