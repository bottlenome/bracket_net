import random
import torch
import torch.utils.data as data


def gen_train_test(num, frac_train=0.3, seed=0):
    # Generate train and test split
    pairs = [[i, j, num] for i in range(num) for j in range(num)]
    random.seed(seed)
    random.shuffle(pairs)
    div = int(frac_train*len(pairs))
    return pairs[:div], pairs[div:]


def create_dataloader(p, batch_size):
    train, test = gen_train_test(p)
    train_result = [[-100, -100, (i+j) % p] for i, j, _ in train]
    test_result = [[-100, -100, (i+j) % p] for i, j, _ in test]
    # concating the train and train result
    train = data.TensorDataset(torch.tensor(train), torch.tensor(train_result))
    test = data.TensorDataset(torch.tensor(test), torch.tensor(test_result))
    # convert train and train_result to DataLoader
    train_loader = data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


if __name__ == "__main__":
    p = 113
    batch_size = 2
    train_loader, test_loader = create_dataloader(p, batch_size)
    print(train_loader)
    print(test_loader)
    for i, (x, y) in enumerate(train_loader):
        print(f"Batch {i} - x: {x}, y: {y}")
    for i, (x, y) in enumerate(test_loader):
        print(f"Batch {i} - x: {x}, y: {y}")
