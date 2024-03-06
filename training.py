import time
from datetime import datetime

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from dataloading import sentence_to_indices


today_now = datetime.today().strftime("%y%m%d-%-H%M")
PAD_token = 0
SOS_token = 1
EOS_token = 2



def min_sec(t):
    """Convert seconds to minutes and seconds"""
    sec = t % 60
    min = (t-sec)/60
    return f"{min:2.0f}:{sec:2.0f}"

def time_remaining(epoch, n_epochs, batch, n_batches, time):
    """Calculate time remaining for training or evaluation"""
    b_comp = epoch * n_batches + batch
    b_rem = n_epochs * n_batches - b_comp
    t_batch = time / b_comp
    t_rem = b_rem * t_batch
    return t_rem

def sentence_to_tensor(lang, sentence):
    """Returns tensor of integer indices for words in "sentence", including EOS_token index"""
    indices = sentence_to_indices(lang, sentence)
    indices.append(EOS_token)
    return torch.tensor(indices, dtype=torch.long).view(1, -1)


def train_epoch(dataloader, encoder, decoder, enc_optimizer, dec_optimizer, loss_fn,
                print_batch, save_batch, enc_path, dec_path, epoch, n_epochs):
    """Train a single epoch and return loss over that epoch"""

    start = time.time()
    total_loss = 0.0
    n_batches = len(dataloader)
    for k, batch in enumerate(dataloader):
        input, target = batch

        # zero gradients
        enc_optimizer.zero_grad()
        dec_optimizer.zero_grad()

        h, h_B_Tx = encoder(input)
        y = decoder(h, h_B_Tx)

        B, T_y, K_y = y.size()
        y_flat = torch.reshape(y, (B*T_y, K_y))
        target_flat = torch.flatten(target)

        loss = loss_fn(y_flat, target_flat)
        loss.backward()

        enc_optimizer.step()
        dec_optimizer.step()

        total_loss += loss.item()

        if (k+1) % print_batch == 0:
            now = time.time()
            elapsed = now-start
            remaining = time_remaining(epoch, n_epochs, k+1, n_batches, elapsed)
            print(f"   batch {k+1} of {n_batches}: {min_sec(elapsed)} elapsed, {min_sec(remaining)} remaining")

        if save_batch > 0:
            if (k+1) % save_batch == 0:
                enc_path_batch = f"{enc_path}b{k+1}.pth"
                dec_path_batch = f"{dec_path}b{k+1}.pth"
                torch.save(encoder.state_dict(), enc_path_batch)
                torch.save(decoder.state_dict(), dec_path_batch)
                print("<MODEL SAVED>")

    return total_loss / n_batches


def train(dataloader, encoder, decoder, enc_optimizer, dec_optimizer, loss_fn,
          save_path, n_epochs=1, print_batch=10, save_batch=0):
    """Train over multiple epochs and save weights at each epoch"""
    
    print("Starting training")
    start = time.time()

    for epoch in range(n_epochs):
        enc_path = f"{save_path}_ENC({today_now})E{epoch+1}"
        dec_path = f"{save_path}_DEC({today_now})E{epoch+1}"

        print("")
        print(f"EPOCH {epoch+1} of {n_epochs}")
        loss = train_epoch(dataloader, encoder, decoder, enc_optimizer, dec_optimizer, loss_fn,
                           print_batch, save_batch, enc_path, dec_path, epoch, n_epochs)
        now = time.time()
        print(f"LOSS = {loss:.4f} ({now-start:.2f}s)")

        enc_path += ".pth"
        dec_path += ".pth"
        torch.save(encoder.state_dict(), enc_path)
        torch.save(decoder.state_dict(), dec_path)
        print("Model saved to:")
        print(enc_path)
        print(dec_path)


def evaluate(dataloader, encoder, decoder):
    """Evaluate model on evaluation dataset and return accuracy"""

    print("Starting evaluation")
    start = time.time()

    n_batches = len(dataloader)
    total_correct = 0
    total_possible = 0
    
    with torch.no_grad():
        for k, batch in enumerate(dataloader):
            input, target = batch
            h, h_B_Tx = encoder(input)
            y = decoder(h, h_B_Tx, evaluate=True)

            correct = torch.sum(torch.where(y == target, 1, 0)).item()
            possible = target.numel()

            # count number of shared zero elements (to remove from count)
            mask_y = torch.where(y != 0, -1, 0)
            mask_target = torch.where(target != 0, -2, 0)
            n_zeros = torch.sum(torch.where(mask_y == mask_target, 1, 0)).item()

            correct -= n_zeros
            possible -= n_zeros

            now = time.time()
            elapsed = now-start
            remaining = time_remaining(0, 1, k+1, n_batches, elapsed)
            print(f"Batch {k+1} of {n_batches}: {correct} out of {possible} words correct")
            print(f"   {min_sec(elapsed)} elapsed, {min_sec(remaining)} remaining")

            total_correct += correct
            total_possible += possible

    print(f"ALL BATCHES: {total_correct} out of {total_possible} words correct")
    return total_correct / total_possible


def translate(pair, in_lang, out_lang, encoder, decoder):
    """Translate single sentence and return sentence as a string"""

    input = sentence_to_tensor(in_lang, pair[0])
    
    h, h_B_Tx = encoder(input)
    y = decoder(h, h_B_Tx, evaluate=True).squeeze()
    print(y.size())

    # convert to string
    words = []
    for idx in y:
        if idx.item() == EOS_token:
            words.append("<EOS>")
            break
        words.append(out_lang.index2word[idx.item()])
    translation = " ".join(words)

    return translation

