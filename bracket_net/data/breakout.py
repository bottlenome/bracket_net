from torch.utils.data.dataloader import DataLoader
from atari.create_dataset import create_dataset, StateActionReturnDataset

def create_dataloader(num_buffers=10,
                      num_steps=100000,
                      trajectories_per_buffer=10,
                      batch_size=128,
                      context_length=30):
    # obss, actions, returns, done_idxs, rtg, timesteps
    # obss [2006, 4, 84, 84]
    # actions [2006]
    # returns [3]
    # done_idxs [2]
    # rtg [2006]
    # timesteps [2007]
    obss, actions, returns, done_idxs, rtgs, timesteps = create_dataset(
        num_buffers=num_buffers,
        num_steps=num_steps, game='Breakout',
        data_dir_prefix='data/',
        trajectories_per_buffer=trajectories_per_buffer)
    
    dataset = StateActionReturnDataset(obss, context_length, actions, done_idxs, rtgs, timesteps)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                             num_workers=4)
    return data_loader


if __name__ == '__main__':
    data_loader = create_dataloader(num_steps=1000, batch_size=11, context_length=60)
    print(len(data_loader))
    for i, (states, actions, rtgs, timesteps) in enumerate(data_loader):
        print(states.shape, actions.shape, rtgs.shape, timesteps.shape)
        print(timesteps)
        if i == 0:
            break