import numpy as np
import os
import random
import torch
from torch import nn
import torch.nn.functional as F
from torchtext import data, datasets, vocab
import tqdm
import json
import yaml
import wandb

from transformer import TransformerClassifier, to_device

NUM_CLS = 2
VOCAB_SIZE = 50_000
SAMPLED_RATIO = 0.2
MAX_SEQ_LEN = 512

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def prepare_data_iter(sampled_ratio=0.2, batch_size=16):
    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
    LABEL = data.Field(sequential=False)
    tdata, _ = datasets.IMDB.splits(TEXT, LABEL)
    # Reduce dataset size
    reduced_tdata, _ = tdata.split(split_ratio=sampled_ratio)
    # Create train and test splits
    train, test = reduced_tdata.split(split_ratio=0.8)
    print('training: ', len(train), 'test: ', len(test))
    TEXT.build_vocab(train, max_size= VOCAB_SIZE - 2)
    LABEL.build_vocab(train)
    train_iter, test_iter = data.BucketIterator.splits((train, test), 
                                                       batch_size=16, 
                                                       device=to_device()
    )

    return train_iter, test_iter

def train():
    with wandb.init() as run:
        sweep_config = wandb.config

        loss_function = nn.CrossEntropyLoss()

        train_iter, test_iter = prepare_data_iter(sampled_ratio=SAMPLED_RATIO, 
                                                batch_size=sweep_config.batch_size
        )


        model = TransformerClassifier(embed_dim=sweep_config.embed_dim, 
                                    num_heads=sweep_config.num_heads,
                                    num_layers=sweep_config.num_layers,
                                    pos_enc=sweep_config.pos_enc,
                                    pool=sweep_config.pool,
                                    dropout=sweep_config.dropout,
                                    max_seq_len=MAX_SEQ_LEN, 
                                    num_tokens=VOCAB_SIZE, 
                                    num_classes=NUM_CLS,
                                    )
        
        if torch.cuda.is_available():
            model = model.to('cuda')

        opt = torch.optim.AdamW(lr=sweep_config["lr"], params=model.parameters(), weight_decay=sweep_config.weight_decay)
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / sweep_config.weight_decay, 1.0))

        val_acc= []
        # training loop
        for e in range(sweep_config.num_epochs):
            print(f'\n epoch {e}')
            model.train()
            for batch in tqdm.tqdm(train_iter):
                opt.zero_grad()
                input_seq = batch.text[0]
                batch_size, seq_len = input_seq.size()
                label = batch.label - 1
                if seq_len > MAX_SEQ_LEN:
                    input_seq = input_seq[:, :MAX_SEQ_LEN]
                out = model(input_seq)
                loss = loss_function(out, label)
                loss.backward()
                # if the total gradient vector has a length > 1, we clip it back down to 1.
                if sweep_config["gradient_clipping"] > 0.0:
                    nn.utils.clip_grad_norm_(model.parameters(), sweep_config["gradient_clipping"])
                opt.step()
                sch.step()

            with torch.no_grad():
                model.eval()
                tot, cor= 0.0, 0.0
                for batch in test_iter:
                    input_seq = batch.text[0]
                    batch_size, seq_len = input_seq.size()
                    label = batch.label - 1
                    if seq_len > MAX_SEQ_LEN:
                        input_seq = input_seq[:, :MAX_SEQ_LEN]
                    out = model(input_seq).argmax(dim=1)
                    tot += float(input_seq.size(0))
                    cor += float((label == out).sum().item())
                acc = cor / tot
                val_acc.append(acc)
                wandb.log({"epoch": e, "val_acc": acc})
                print(f'-- {"validation"} accuracy {acc:.3}')
    wandb.finish()

def main():
    with open("configs/sweep.yaml", "r") as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep_config, project="transformer_text_classification")
    wandb.agent(sweep_id, function=train, count = 3)
    

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]= str(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(f"Model will run on {device}")
    set_seed(seed=1)
    main()
