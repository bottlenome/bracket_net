from torch.utils.data.dataloader import DataLoader
from .atari.create_dataset import create_dataset, StateActionReturnDataset

def create_dataloader(data_name='Breakout',
                      val_test_rate=0.1,
                      batch_size=128,
                      num_buffers=49,
                      num_steps=100000,
                      trajectories_per_buffer=10,
                      context_length=30):
    # obss, actions, returns, done_idxs, rtg, timesteps
    # obss [2006, 4, 84, 84]
    # actions [2006]
    # returns [3]
    # done_idxs [2]
    # rtg [2006]
    # timesteps [2007]
    obss, actions, _, done_idxs, rtgs, timesteps = create_dataset(
        num_buffers=num_buffers,
        num_steps=num_steps,
        game=data_name,
        data_dir_prefix='data/',
        trajectories_per_buffer=trajectories_per_buffer)

    dataset = StateActionReturnDataset(obss, context_length, actions, done_idxs, rtgs, timesteps)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                              num_workers=4)
    obss, actions, _, done_idxs, rtgs, timesteps = create_dataset(
        num_buffers=10,
        num_steps=10000,
        game=data_name,
        data_dir_prefix='data/',
        trajectories_per_buffer=int(trajectories_per_buffer * val_test_rate), episode="2")
    dataset = StateActionReturnDataset(obss, context_length, actions, done_idxs, rtgs, timesteps)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            pin_memory=True, num_workers=4)
    obss, actions, _, done_idxs, rtgs, timesteps = create_dataset(
        num_buffers=10,
        num_steps=10000,
        game=data_name,
        data_dir_prefix='data/',
        trajectories_per_buffer=int(trajectories_per_buffer * val_test_rate), episode="3")
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             pin_memory=True, num_workers=4)
    print(f"train: {len(train_loader)}, val: {len(val_loader)}, test: {len(test_loader)}")
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    data_loader, _, _ = create_dataloader(num_steps=1000, batch_size=11, context_length=60)
    print(len(data_loader))
    for i, (states, actions, rtgs, timesteps) in enumerate(data_loader):
        print(states.shape, actions.shape, rtgs.shape, timesteps.shape)
        print(timesteps)
        if i == 0:
            break