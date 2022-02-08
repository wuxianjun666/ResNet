import argparse
import json
import os
import time
import torch
import torchvision
from torch import nn,optim
from torchvision import transforms
from torch.utils.data import DataLoader
import model

device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser("""Image classifical!""")

parser.add_argument('--path',type=str,default='data/cifar100/',
                    help="""image dir path default: 'data/cifar100/'.""")
parser.add_argument('--epochs',type=int,default=50,
                    help="""Epoch default:50.""")
parser.add_argument('--batch_size',type=int,default=1024,
                    help="""Batch_size default:256.""")
parser.add_argument('--lr',type=float,default=0.0001,
                    help="""learing_rate. Default=0.0001""")
parser.add_argument('--num_classes',type=int,default=10,
                    help="""num classes""")
parser.add_argument('--model_path',type=str,default='models/',
                    help="""Save model path""")
parser.add_argument('--model_name',type=str,default='cifar100.pth',
                    help="""Model name.""")
parser.add_argument('--display_epoch',type=int,default=1)

args = parser.parse_args()

if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomHorizontalFlip(p=0.75),
    transforms.RandomCrop(24),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_datasets = torchvision.datasets.CIFAR100(root=args.path,
                                               transform=transform,
                                               download=True,
                                               train=True)
train_loader = DataLoader(dataset=train_datasets,
                          batch_size=args.batch_size,
                          shuffle=True)

test_datasets = torchvision.datasets.CIFAR100(root=args.path,
                                              transform=transform,
                                              download=True,
                                              train=False)
test_loader = DataLoader(dataset=test_datasets,
                         batch_size=args.batch_size,
                         shuffle=False)

flower = train_datasets.class_to_idx
class_dic =dict((k,v) for k,v in flower.items())
json_str =json.dumps(class_dic)
with open('class_indices.json','w') as json_file:
    json_file.write(json_str)

def train():
    print(f"Train numbers:{len(train_datasets)}")
    net = model.ResNet101()
    loss_function =nn.CrossEntropyLoss()
    optimizer =optim.Adam(net.parameters(),lr = args.lr)

    best_acc =0.0
    save_path = './ResNet.pth'
    start = time.time()
    for epoch in range(args.epochs):
        net.train()
        for step,data in enumerate(train_loader,start=0):
            image,label = data
            output = net(image.to(device))
            optimizer.zero_grad()
            loss = loss_function(output,label.to(device))
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch}/{args.epochs}], "
                  f"Loss: {loss.item():.8f}, "
                  f"Time: {(time.time()-start) * args.display_epoch:.1f}sec!")

            net.eval()

            with torch.no_grad():
                acc = 0.0
                for test_data in test_loader:
                    test_image,test_label = test_data
                    out =  net(test_image.to(device))
                    pred = torch.max(out,dim=1)[1]

                    acc += torch.eq(pred,test_label.to(device)).sum().item()
                test_acc = acc / len(test_datasets)
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(net.state_dict(),save_path)
        print('[epoch %d]  test_accuracy: %.3f' % (epoch+1,test_acc))

    print('Training Finished!')

if __name__ == '__main__':
    train()











