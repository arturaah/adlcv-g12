import numpy as np
import os
import random
import torch
from torch import nn
import torch.nn.functional as F
import tqdm
import wandb
import torch
import torchvision
import torchvision.transforms as transforms
from vit import ViT
import yaml

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def select_two_classes_from_cifar10(dataset, classes):
    idx = (np.array(dataset.targets) == classes[0]) | (np.array(dataset.targets) == classes[1])
    dataset.targets = np.array(dataset.targets)[idx]
    dataset.targets[dataset.targets==classes[0]] = 0
    dataset.targets[dataset.targets==classes[1]] = 1
    dataset.targets= dataset.targets.tolist()  
    dataset.data = dataset.data[idx]
    return dataset

def prepare_dataloaders(batch_size, classes=[3, 7]):
    # TASK: Experiment with data augmentation
    train_transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    test_transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=test_transform)

    # select two classes 
    trainset = select_two_classes_from_cifar10(trainset, classes=classes)
    testset = select_two_classes_from_cifar10(testset, classes=classes)

    # reduce dataset size
    trainset, _ = torch.utils.data.random_split(trainset, [5000, 5000])
    testset, _ = torch.utils.data.random_split(testset, [1000, 1000])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=False
    )
    return trainloader, testloader, trainset, testset


def train():
    with wandb.init() as run:
        sweep_config = wandb.config
        run.name= f"layers_{sweep_config.num_layers}, heads_{sweep_config.num_heads}, pos_enc_{sweep_config.pos_enc}, pool_{sweep_config.pool}"

        loss_function = nn.CrossEntropyLoss()

        train_iter, test_iter, _, _ = prepare_dataloaders(batch_size=sweep_config.batch_size)

        model = ViT(image_size=(32,32), patch_size=(4,4), channels=sweep_config.channels, 
                    embed_dim=sweep_config.embed_dim, num_heads=sweep_config.num_heads, num_layers=sweep_config.num_layers,
                    pos_enc=sweep_config.pos_enc, pool=sweep_config.pool, dropout=sweep_config.dropout, fc_dim=None, 
                    num_classes=sweep_config.num_classes
        )

        if torch.cuda.is_available():
            model = model.to('cuda')

        opt = torch.optim.AdamW(lr=sweep_config.lr, params=model.parameters(), weight_decay=sweep_config.weight_decay)
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / sweep_config.warmup_steps, 1.0))
        
        # training loop
        best_val_loss = 1e10
        for e in range(sweep_config.num_epochs):
            print(f'\n epoch {e}')
            model.train()
            train_loss = 0
            for image, label in tqdm.tqdm(train_iter):
                if torch.cuda.is_available():
                    image, label = image.to('cuda'), label.to('cuda')
                opt.zero_grad()
                out = model(image)
                loss = loss_function(out, label)
                loss.backward()
                train_loss += loss.item()
                # if the total gradient vector has a length > 1, we clip it back down to 1.
                if sweep_config.gradient_clipping > 0.0:
                    nn.utils.clip_grad_norm_(model.parameters(), sweep_config.gradient_clipping)
                opt.step()
                sch.step()

            train_loss/=len(train_iter)

            val_loss = 0
            with torch.no_grad():
                model.eval()
                tot, cor= 0.0, 0.0
                for image, label in test_iter:
                    if torch.cuda.is_available():
                        image, label = image.to('cuda'), label.to('cuda')
                    out = model(image)
                    loss = loss_function(out, label)
                    val_loss += loss.item()
                    out = out.argmax(dim=1)
                    tot += float(image.size(0))
                    cor += float((label == out).sum().item())
                acc = cor / tot
                val_loss /= len(test_iter)
                wandb.log({"epoch": e, "val acc": acc, "val_loss" :val_loss})

                print(f'-- train loss {train_loss:.3f} -- validation accuracy {acc:.3f} -- validation loss: {val_loss:.3f}')
                if val_loss <= best_val_loss:
                    torch.save(model.state_dict(), 'model.pth')
                    best_val_loss = val_loss
    wandb.finish()
def main():
    with open("configs/image_sweep.yaml", "r") as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep_config, project="image_classification")
    wandb.agent(sweep_id, function=train)

if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"]= str(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(f"Model will run on {device}")
    set_seed(seed=1)
    main()
