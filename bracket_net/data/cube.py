from .cube_model.face import FaceCube
from .cube_model.defs import N_MOVE
from .cube_model.enums import Move
from .cube_model import cubie
from .cube_model.coord import CoordCube
from .cube_model import moves as mv

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterableWrapper
from torch.nn.utils.rnn import pad_sequence
import pickle as pkl
import os
from torch.utils.data import random_split

from functools import cache

class RubicDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.data[idx])
        return self.data[idx]

def R222ShortestAll(transform=None, size=None):
    """ Load R222ShortestAll.pkl that contains all shortest moves of 2x2x2 cube
        data: (face, shortest_moves)
    """
    data = pkl.load(open('./data/R222ShortestAll.pkl', 'rb'))
    if size is not None:
        data = data[:size]
    return RubicDataset(data, transform=transform)


class RubicDFSDataSet(Dataset):
    def __init__(self, file_dir="./data/R222DFS/", max_size=None):
        self.file_dir = file_dir
        self.file_list = sorted([os.path.join(file_dir, f) for f in os.listdir(file_dir) if f.endswith('.pkl')])
        self.cache = {}
        self.max_size = max_size

        # 最初のファイルをロードしてデータ数を推測
        first_file_path = self.file_list[0]
        with open(first_file_path, 'rb') as file:
            first_data = pkl.load(file)
            self.data_length = len(first_data)

        # max_sizeに基づいて実際のデータ数を計算
        if self.max_size is not None:
            self.total_data_points = min(self.max_size, len(self.file_list) * self.data_length)
        else:
            self.total_data_points = len(self.file_list) * self.data_length

    def __len__(self):
        return self.total_data_points

    def __getitem__(self, idx):
        if idx >= self.total_data_points:
            raise IndexError("Index out of range")

        file_idx = idx // self.data_length
        data_idx = idx % self.data_length

        if file_idx in self.cache:
            data = self.cache[file_idx]
        else:
            file_path = self.file_list[file_idx]
            with open(file_path, 'rb') as file:
                data = pkl.load(file)
                self.cache[file_idx] = data

        return self.cache[file_idx][data_idx]


def char2move(char):
    if char == "U1":
        return Move.U1
    elif char == "U2":
        return Move.U2
    elif char == "U3":
        return Move.U3
    elif char == "R1":
        return Move.R1
    elif char == "R2":
        return Move.R2
    elif char == "R3":
        return Move.R3
    elif char == "F1":
        return Move.F1
    elif char == "F2":
        return Move.F2
    elif char == "F3":
        return Move.F3
    else:
        raise ValueError("Invalid move char: {}".format(char))


def char2move_int(char):
    if char == "U1":
        return 1
    elif char == "U2":
        return 2
    elif char == "U3":
        return 3
    elif char == "R1":
        return 4
    elif char == "R2":
        return 5
    elif char == "R3":
        return 6
    elif char == "F1":
        return 7
    elif char == "F2":
        return 8
    elif char == "F3":
        return 9
    else:
        raise ValueError("Invalid move char: {}".format(char))


def face_str2int(face_str):
    ret = []
    for c in face_str:
        if c == 'U':
            ret.append(1)
        elif c == 'R':
            ret.append(2)
        elif c == 'F':
            ret.append(3)
        elif c == 'D':
            ret.append(4)
        elif c == 'L':
            ret.append(5)
        elif c == 'B':
            ret.append(6)
        else:
            raise ValueError("Invalid face char: {}".format(c))
    return ret

def face_int2str(face_int):
    ret = ""
    for i in face_int:
        if i == 1:
            ret += 'U'
        elif i == 2:
            ret += 'R'
        elif i == 3:
            ret += 'F'
        elif i == 4:
            ret += 'D'
        elif i == 5:
            ret += 'L'
        elif i == 6:
            ret += 'B'
        else:
            raise ValueError("Invalid face int: {}".format(i))
    return ret

def make_state_and_solve_state(batch):
    FACE = 0
    MOVE = 1
    inputs = [face_str2int(item[FACE]) for item in batch]
    targets = []
    for item in batch:
        faces = [[0] * 24]
        fc = FaceCube()
        s = fc.from_string(item[FACE])
        if s is not True:
            raise ValueError("Error in facelet cube")
        cc = fc.to_cubie_cube()
        s = cc.verify()
        if s != cubie.CUBE_OK:
            raise ValueError("Error in cubie cube")
        co_cube = CoordCube(cc)
        for i in range(0, len(item[MOVE]), 2):
            m = char2move(item[MOVE][i:i+2])
            co_cube.corntwist = mv.corntwist_move[N_MOVE * co_cube.corntwist + m]
            co_cube.cornperm = mv.cornperm_move[N_MOVE * co_cube.cornperm + m]
            cc = cubie.CubieCube()
            cc.set_corners(co_cube.cornperm)
            cc.set_cornertwist(co_cube.corntwist)
            faces.append(face_str2int(cc.to_facelet_cube().to_string()))
        targets.append(torch.tensor(faces))
    targets = pad_sequence(targets, batch_first=True, padding_value=0)
    return torch.tensor(inputs), targets


def make_state_and_distance(batch):
    FACE = 0
    MOVE = 1
    inputs = [face_str2int(item[FACE]) for item in batch]
    targets = [len(item[MOVE]) // 2 for item in batch]
    return torch.tensor(inputs), torch.tensor(targets)


def make_state_and_action(batch):
    FACE = 0
    MOVE = 1
    inputs = [face_str2int(item[FACE]) for item in batch]
    targets = [char2move_int(item[MOVE][0:2]) - 1 for item in batch]

    return torch.tensor(inputs), torch.tensor(targets)


def make_state_action_sequence(batch):
    FACE = 0
    MOVE = 1
    actions = []
    for item in batch:
        sequence = []
        for i in range(0, len(item[MOVE]), 2):
            sequence.append(char2move_int(item[MOVE][i:i+2]))
        actions.append(torch.tensor(sequence))
    actions = pad_sequence(actions, batch_first=True, padding_value=0)

    states = []
    for item in batch:
        sequence = []
        sequence.append(face_str2int(item[FACE]))
        fc = FaceCube()
        s = fc.from_string(item[FACE])
        if s is not True:
            raise ValueError("Error in facelet cube")
        cc = fc.to_cubie_cube()
        s = cc.verify()
        if s != cubie.CUBE_OK:
            raise ValueError("Error in cubie cube")
        co_cube = CoordCube(cc)
        for i in range(0, len(item[MOVE]), 2):
            m = char2move(item[MOVE][i:i+2])
            co_cube.corntwist = mv.corntwist_move[N_MOVE * co_cube.corntwist + m]
            co_cube.cornperm = mv.cornperm_move[N_MOVE * co_cube.cornperm + m]
            cc = cubie.CubieCube()
            cc.set_corners(co_cube.cornperm)
            cc.set_cornertwist(co_cube.corntwist)
            sequence.append(face_str2int(cc.to_facelet_cube().to_string()))
        states.append(torch.tensor(sequence))
    states = pad_sequence(states, batch_first=True, padding_value=0)
    return states, actions


def make_reward_state_action_sequence(batch):
    MOVE = 1
    states, actions = make_state_action_sequence(batch)
    rewards = []
    for item in batch:
        sequence = []
        for i in range(0, len(item[MOVE]) + 2, 2):
            rtgs = 100 - len(item[MOVE]) // 2 + i // 2
            sequence.append(rtgs)
        rewards.append(torch.tensor(sequence))
    rewards = pad_sequence(rewards, batch_first=True, padding_value=0)
    return rewards, states, actions


@cache
def get_solved_state():
    fc = FaceCube()
    return torch.tensor(face_str2int(fc.to_string()))


def get_moves(batch):
    MOVE = 1
    moves_batch = []
    for item in batch:
        moves = []
        move = item[MOVE]
        for i in range(0, len(move), 2):
            moves.append(char2move_int(move[i:i+2]))
        moves_batch.append(torch.tensor(moves))
    moves_batch = pad_sequence(moves_batch, batch_first=True, padding_value=0)
    return moves_batch


def BaseLoader(collate_fn, val_test_rate=0.1, batch_size=32, size=None):
    data = R222ShortestAll(size=size)

    train_data_start = 1 # ignore the first data because it is solved state
    train_data_end = int(len(data)*(1 - 2*val_test_rate))
    train_dataloader = IterableWrapper(data[train_data_start:train_data_end])
    train_dataloader = train_dataloader.batch(batch_size=batch_size, drop_last=True)
    train_dataloader = train_dataloader.collate(collate_fn=collate_fn)
    train_dataloader = train_dataloader.in_memory_cache(size=500000)
    train_dataloader = train_dataloader.shuffle(buffer_size=500000)

    val_data_start = train_data_end
    val_data_end = train_data_end + int(len(data)*val_test_rate)
    val_dataloader = IterableWrapper(data[val_data_start:val_data_end])
    val_dataloader = val_dataloader.batch(batch_size=batch_size, drop_last=True)
    val_dataloader = val_dataloader.collate(collate_fn=collate_fn)
    val_dataloader = val_dataloader.in_memory_cache(size=100000)

    test_data_start = val_data_end
    test_data_end = val_data_end + int(len(data)*val_test_rate)
    test_dataloader = IterableWrapper(data[test_data_start:test_data_end])
    test_dataloader = test_dataloader.batch(batch_size=batch_size, drop_last=True)
    test_dataloader = test_dataloader.collate(collate_fn=collate_fn)
    test_dataloader = test_dataloader.in_memory_cache(size=100000)

    return train_dataloader, val_dataloader, test_dataloader


def NOPLoader(val_test_rate=0.1, batch_size=32, size=None):
    def collate_fn(batch):
        FACE = 0
        MOVE = 1

        inputs = [face_str2int(item[FACE]) + [0] for item in batch]
        targets = []

        for item in batch:
            moves = []
            for i in range(0, len(item[MOVE]), 2):
                moves.append(int(char2move(item[MOVE][i:i+2]) + 1 + 6))
            targets.append(torch.tensor(moves))
        targets = pad_sequence(targets, batch_first=True, padding_value=0)
        return torch.tensor(inputs), targets 
    return BaseLoader(collate_fn, val_test_rate, batch_size, size)


def StateLoader(val_test_rate=0.1, batch_size=32, size=None):
    return BaseLoader(make_state_and_solve_state, val_test_rate, batch_size, size)


def NumLoader(val_test_rate=0.1, batch_size=32, size=None):
    # make states as inputs, and move num as targets
    def collate_fn(batch):
        MOVE = 1
        start_state, solve_states = make_state_and_solve_state(batch)
        start_state = start_state.view(start_state.size(0), 1, -1)
        inputs = torch.cat((start_state, solve_states[:, 1:, :]), dim=1)
        inputs[:, -1] = get_solved_state()
        targets = []
        for item in batch:
            targets.append(torch.arange(len(item[MOVE]) // 2, 0, -1))
        targets = pad_sequence(targets, batch_first=True, padding_value=0)
        return inputs, targets
    return BaseLoader(collate_fn, val_test_rate, batch_size, size)


def StateLoader2(val_test_rate=0.1, batch_size=32, size=None):
    def collate_fn(batch):
        start_state, solve_states = make_state_and_solve_state(batch)
        inputs = solve_states[:]
        inputs[:, 0] = start_state
        targets = solve_states[:, 1:]
        return inputs, targets
    return BaseLoader(collate_fn, val_test_rate, batch_size, size)


def AllLoader(val_test_rate=0.1, batch_size=32, size=None):
    def collate_fn(batch):
        moves = get_moves(batch)
        start_state, solve_states = make_state_and_solve_state(batch)
        inputs = solve_states[:]
        inputs[:, 0] = start_state
        targets = solve_states[:, 1:]
        return inputs, [targets, moves]
    return BaseLoader(collate_fn, val_test_rate, batch_size, size)


def StateDistanceLoader(val_test_rate=0.1, batch_size=32, size=None):
    return BaseLoader(make_state_and_distance, val_test_rate, batch_size, size)


def StateNextActionLoader(val_test_rate=0.1, batch_size=32, size=None):
    return BaseLoader(make_state_and_action, val_test_rate, batch_size, size)


def StateActionLoader(val_test_rate=0.1, batch_size=32, size=None):
    return BaseLoader(make_state_action_sequence, val_test_rate, batch_size, size)

def RewardStateActionLoader(val_test_rate=0.1, batch_size=32, size=None):
    return BaseLoader(make_reward_state_action_sequence, val_test_rate, batch_size, size)

DFS_STACK_WIDTH = 20
DFS_STACK_HEIGHT = 20
def RubicDFSLoader(val_test_rate=0.1, batch_size=32, size=None):
    data = RubicDFSDataSet(max_size=size)

    train_size = int(len(data) * (1 - 2 * val_test_rate))
    val_size = int(len(data) * val_test_rate)
    test_size = int(len(data) * val_test_rate)

    train, val, test = random_split(data, [train_size, val_size, test_size])

    def parse_and_tensorize(batch_data):
        states = []
        dones = []
        stacks = []
        histories = []

        for item in batch_data:
            states.append(torch.tensor(item["state"], dtype=torch.float32))
            dones.append(torch.tensor(item["done"], dtype=torch.float32))

            stack_items = []
            for stack_item in item["stack"]:
                perm, twist, sofar, depth = stack_item
                # convert sofar str move to int move
                sofar = [char2move_int(m) for m in sofar]
                stack_items.append(torch.tensor([perm, twist, depth] + sofar, dtype=torch.float32))
            stack_items = pad_sequence(stack_items, batch_first=True, padding_value=0)
            stacks.append(stack_items)
            histoy_items = [char2move_int(m) for m in item["history"]]
            histories.append(torch.tensor(histoy_items, dtype=torch.float32))

        states = torch.stack(states)
        dones = torch.stack(dones)
        # 2d padding for stack list
        max_h = max([s.shape[0] for s in stacks])
        if max_h < DFS_STACK_HEIGHT:
            max_h = DFS_STACK_HEIGHT
        else:
            raise ValueError("stack height is too large")
        max_w = max([s.shape[1] for s in stacks])
        if max_w < DFS_STACK_WIDTH:
            max_w = DFS_STACK_WIDTH
        else:
            raise ValueError("stack width is too large")
        for i in range(len(stacks)):
            s_i = torch.zeros(max_h, max_w, dtype=torch.float32)
            s_i[:stacks[i].shape[0], :stacks[i].shape[1]] = stacks[i]
            stacks[i] = s_i
        padded_stacks = torch.stack(stacks)
        padded_histories = pad_sequence(histories, batch_first=True, padding_value=0)
        return states, dones, padded_stacks, padded_histories

    def collate_fn(batch):
        states = []
        dones = []
        stacks = []
        histories = []
        for item in batch:
            state, done, stack, history = parse_and_tensorize(item)
            states.append(state)
            dones.append(done)
            stacks.append(stack)
            histories.append(history)
        states = pad_sequence(states, batch_first=True, padding_value=-1)
        dones = pad_sequence(dones, batch_first=True, padding_value=-1)

        # 3d padding for stack list
        max_c = max([s.shape[0] for s in stacks])
        max_h = max([s.shape[1] for s in stacks])
        max_w = max([s.shape[2] for s in stacks])
        for i in range(len(stacks)):
            s_i = torch.zeros(max_c, max_h, max_w, dtype=torch.float32)
            s_i[:stacks[i].shape[0], :stacks[i].shape[1], :stacks[i].shape[2]] = stacks[i]
            stacks[i] = s_i
        stacks = torch.stack(stacks)

        # 2d padding for histories list
        max_h = max([h.shape[0] for h in histories])
        max_w = max([h.shape[1] for h in histories])
        for i in range(len(histories)):
            h_i = torch.zeros(max_h, max_w, dtype=torch.float32)
            h_i[:histories[i].shape[0], :histories[i].shape[1]] = histories[i]
            histories[i] = h_i
        histories = torch.stack(histories)

        return states, dones, stacks, histories

    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_dataloader, val_dataloader, test_dataloader


def create_dataloader(loder_name, val_test_rate, batch_size, size=None):
    if loder_name == "NOPLoader":
        return NOPLoader(val_test_rate=val_test_rate, batch_size=batch_size, size=size)
    elif loder_name == "StateLoader":
        return StateLoader(val_test_rate=val_test_rate, batch_size=batch_size, size=size)
    elif loder_name == "NumLoader":
        return NumLoader(val_test_rate=val_test_rate, batch_size=batch_size, size=size)
    elif loder_name == "StateLoader2":
        return StateLoader2(val_test_rate=val_test_rate, batch_size=batch_size, size=size)
    elif loder_name == "AllLoader":
        return AllLoader(val_test_rate=val_test_rate, batch_size=batch_size, size=size)
    elif loder_name == "StateDistanceLoader":
        return StateDistanceLoader(val_test_rate=val_test_rate, batch_size=batch_size, size=size)
    elif loder_name == "StateNextActionLoader":
        return StateNextActionLoader(val_test_rate=val_test_rate, batch_size=batch_size, size=size)
    elif loder_name == "StateActionLoader":
        return StateActionLoader(val_test_rate=val_test_rate, batch_size=batch_size, size=size)
    elif loder_name == "RewardStateActionLoader":
        return RewardStateActionLoader(val_test_rate=val_test_rate, batch_size=batch_size, size=size)
    elif loder_name == "RubicDFSLoader":
        return RubicDFSLoader(val_test_rate=val_test_rate, batch_size=batch_size, size=size)
    else:
        raise ValueError("Invalid loder_name: {}".format(loder_name))


if __name__ == '__main__':
    data = R222ShortestAll()
    print("data size", len(data))
    data = R222ShortestAll(size=100000)
    print("data size", len(data))
    print("example data[1]", data[1])

    train_dataloader = DataLoader(data[:3000000], batch_size=4, shuffle=True)
    for i in train_dataloader:
        print("batch_size", len(i))
        print(i[0])
        break
    print("RubicDFSDataSet")
    d = RubicDFSDataSet(max_size=1000)
    print("data size", len(d))
    print("example data[1]", d[1])

    print("RubicDFSLoader")
    data_loader, _, _ = RubicDFSLoader(0.1, 10, size=1000)

    for i in data_loader:
        print("batch_size", len(i))
        states, dones, stacks, histories = i
        print("states.shape", states.shape)
        print("dones.shape", dones.shape)
        print("stacks.shape", stacks.shape)
        print("histories.shape", histories.shape)
        """"
        print("states[0]", states[0])
        print("dones[0]", dones[0])
        print("stacks[0]", stacks[0])
        print("histories[0]", histories[0])
        """
        break

    print("NOPLoader")
    train_dataloader, val_dataloader, test_dataloader = create_dataloader("NOPLoader", 0.1, 10, size=1000)
    for i in train_dataloader:
        print("src, tgt", len(i))
        print("src.shape", i[0].shape)
        print("tgt.shape", i[1].shape)
        print("src[0]", i[0][0])
        print("src[0] max:", i[0][0].max())
        print("tgt[0]", i[1][0])
        break

    print("StateLoader")
    train_dataloader, val_dataloader, test_dataloader = create_dataloader("StateLoader", 0.1, 10, size=1000)
    for i in train_dataloader:
        print("src, tgt", len(i))
        print("src.shape", i[0].shape)
        print("tgt.shape", i[1].shape)
        print("src[0]", i[0][0])
        print("src[0] max:", i[0][0].max())
        print("tgt[0]", i[1][0])
        break

    print("NumLoader")
    train_dataloader, val_dataloader, test_dataloader = create_dataloader("NumLoader", 0.1, 10, size=1000)
    j = 0
    for i in train_dataloader:
        if j == 0:
            print("src, tgt", len(i))
            print("src.shape", i[0].shape)
            print("tgt.shape", i[1].shape)
            print("src[0]", i[0][0])
            print("src[0] max:", i[0][0].max())
            print("src[-1]", i[0][0, -1])
            print("tgt[0]", i[1][0])
            j += 1
    j = 0
    for i in train_dataloader:
        if j == 0:
            print("src, tgt", len(i))
            print("src.shape", i[0].shape)
            print("tgt.shape", i[1].shape)
            print("src[0]", i[0][0])
            print("src[0] max:", i[0][0].max())
            print("tgt[0]", i[1][0])
            j += 1
    print(get_solved_state())

    print("StateLoader2")
    train_dataloader, val_dataloader, test_dataloader = create_dataloader("StateLoader2", 0.1, 10, size=1000)
    for i in train_dataloader:
        print("src, tgt", len(i))
        print("src.shape", i[0].shape)
        print("tgt.shape", i[1].shape)
        print("src[0]", i[0][0])
        print("src[0] max:", i[0][0].max())
        print("tgt[0]", i[1][0])
        break

    print("AllLoader")
    train_dataloader, val_dataloader, test_dataloader = create_dataloader("AllLoader", 0.1, 10, size=1000)
    zero_rate = 0.
    TGT = 1
    MOVES = 1
    for index, i in enumerate(train_dataloader):
        zero_rate += (i[:][TGT][MOVES] == 0).sum().item() / (i[TGT][MOVES].size(0) * i[TGT][MOVES].size(1))
        if index == 0:
            print("src, tgt", len(i))
            print("src.shape", i[0].shape)
            print("tgt[0].shape", i[TGT][0].shape)
            print("tgt[1].shape", i[TGT][MOVES].shape)
            print("src[0]", i[0][0])
            print("src[0] max:", i[0][0].max())
            print("tgt[0][0]", i[TGT][0][0])
            print("tgt[1][0]", i[TGT][MOVES][0])
            print("zero_rate", zero_rate)
    print("zero_rate", zero_rate / len(train_dataloader))

    print("StateDistanceLoader")
    train_dataloader, val_dataloader, test_dataloader = create_dataloader("StateDistanceLoader", 0.1, 10, size=1000)
    for i in train_dataloader:
        print("src, tgt", len(i))
        print("src.shape", i[0].shape)
        print("tgt.shape", i[1].shape)
        print("src", i[0])
        print("src[0]", i[0][0])
        print("src[0] max:", i[0][0].max())
        print("tgt", i[1])
        break

    print("StateNextActionLoader")
    train_dataloader, val_dataloader, test_dataloader = create_dataloader("StateNextActionLoader", 0.1, 10, size=1000)
    for i in train_dataloader:
        print("src, tgt", len(i))
        print("src.shape", i[0].shape)
        print("tgt.shape", i[1].shape)
        print("src[0]", i[0][0])
        print("src[0] max:", i[0][0].max())
        print("tgt[0]", i[1][0])
        break
    
    print("StateActionLoader")
    train_dataloader, val_dataloader, test_dataloader = create_dataloader("StateActionLoader", 0.1, 10, size=1000)
    for i in train_dataloader:
        print("state, action", len(i))
        print("state.shape", i[0].shape)
        print("action.shape", i[1].shape)
        print("state[0]", i[0][0])
        print("action[0]", i[1][0])
        break

    print("RewardStateActionLoader")
    train_dataloader, val_dataloader, test_dataloader = create_dataloader("RewardStateActionLoader", 0.1, 10, size=1000)
    for i in train_dataloader:
        print("reward, state, action", len(i))
        print("reward.shape", i[0].shape)
        print("state.shape", i[1].shape)
        print("action.shape", i[2].shape)
        print("reward[0]", i[0][0])
        print("state[0]", i[1][0])
        print("action[0]", i[2][0])
        break

