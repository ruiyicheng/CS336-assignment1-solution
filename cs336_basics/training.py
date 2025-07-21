import numpy as np
import torch
import os
import typing
#import tikz
def data_loading(x,batch_size, context_length, device = None):
    lenx = len(x)
    start_of_sample = np.random.randint(lenx-context_length,size = batch_size)
    end_of_sample = start_of_sample + context_length
    train = []
    test = []
    for s,e in zip(start_of_sample,end_of_sample):
        train.append(x[s:e])
        test.append(x[s+1:e+1])
        #print(x[s:e])
    train = np.array(train).astype(np.int64)
    test = np.array(test).astype(np.int64)

    train_torch = torch.tensor(train,device = device)#.to(torch.UInt16)
    labels_torch = torch.tensor(test,device = device)#.to(torch.UInt16)
    return train_torch,labels_torch



def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    print("Saving iteration:",iteration)
    model_par = model.state_dict()
    optimizer_par = optimizer.state_dict()
    par_dict = {'model':model_par, 'optimizer':optimizer_par,'iter':iteration}
    torch.save(par_dict, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],model: torch.nn.Module,optimizer: torch.optim.Optimizer):
    par_dict = torch.load(src)
    model.load_state_dict(par_dict['model'])
    optimizer.load_state_dict(par_dict['optimizer'])
    return par_dict["iter"]



def train(d_model: int, num_heads: int, d_ff: int, vocab_size: int, num_layers: int,max_seq_len : int,theta : float,data_path = "/Users/ruiyicheng/Documents/code/python/CS336/CS336-assignment1-solution/cs336_basics/tokenized_tinystories_train.npy",total_tokens_train = 40000000,batch_size = 12,device = None):
    from transformer_modules import transformer_lm
    from optimizer import AdamW
    from optimizer import cross_entropy

    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    
    model = transformer_lm(d_model=d_model, num_heads=num_heads, d_ff=d_ff, vocab_size=vocab_size, num_layers=num_layers, max_seq_len=max_seq_len, theta=theta,device = device)
    optimizer = AdamW(model.parameters(), lr=1e-2, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    model = model.to(device)
    #compile
    #model = torch.compile(model)
    model = torch.compile(model, backend="aot_eager")
    i = 0 
    total_tokens = 0
    while True:
        optimizer.zero_grad()
        train, labels = data_loading(data, batch_size=batch_size, context_length=max_seq_len, device=model.device)
        #print(train, labels)
        total_tokens += train.shape[0] * train.shape[1]
        logits = model(train)
        #print(torch.sum(logits,axis = -1))
        loss = cross_entropy(logits,labels)
        print("Iteration:", i, "Loss:", loss.item(),'total tokens:', total_tokens)
        loss.backward()
        optimizer.step()
        i += 1
        if i % 1000 == 0:
            # create dir named after input param
            os.makedirs("checkpoints", exist_ok=True)
            name_this_model = f"d_model_{d_model}_num_heads_{num_heads}_d_ff_{d_ff}_vocab_size_{vocab_size}_num_layers_{num_layers}_max_seq_len_{max_seq_len}_theta_{theta}"
            os.makedirs(f"checkpoints/{name_this_model}", exist_ok=True)
            save_checkpoint(model, optimizer, i, f"checkpoints/{name_this_model}/checkpoint_{i}.pt")
        if total_tokens >= total_tokens_train:
            print("Total tokens reached:", total_tokens)
            save_checkpoint(model, optimizer, i, f"checkpoints/{name_this_model}/checkpoint_{i}.pt")
            break

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a transformer model.")
    parser.add_argument("--d_model", type=int, default=512, help="Dimension of the model.")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of attention heads.")
    parser.add_argument("--d_ff", type=int, default=1344, help="Dimension of the feedforward layer.")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Size of the vocabulary.")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers.")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Maximum sequence length.")
    parser.add_argument("--theta", type=float, default=0.5, help="Theta parameter for the model.")
    args = parser.parse_args()
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    #device = "cpu"
    train(args.d_model, args.num_heads, args.d_ff, args.vocab_size, args.num_layers, args.max_seq_len, args.theta, device=device)