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

    data = R222ShortestAll(size=size)
    train_data_start = 1 # ignore the first data because it is solved state
    train_data_end = int(len(data)*(1 - 2*val_test_rate))
    train_dataloader = DataLoader(data[train_data_start:train_data_end],
                                  batch_size=batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn)
    val_data_start = train_data_end
    val_data_end = train_data_end + int(len(data)*val_test_rate)
    val_dataloader = DataLoader(data[val_data_start:val_data_end],
                                batch_size=batch_size,
                                shuffle=False,
                                collate_fn=collate_fn)
    test_data_start = val_data_end
    test_data_end = val_data_end + int(len(data)*val_test_rate)
    test_dataloader = DataLoader(data[test_data_start:test_data_end],
                                 batch_size=batch_size,
                                 shuffle=False,
                                 collate_fn=collate_fn)
    return train_dataloader, val_dataloader, test_dataloader

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
    targets = [char2move_int(item[MOVE][0:2]) for item in batch]

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
    actions = pad_sequence(torch.tensor(actions), batch_first=True, padding_value=0)
    return torch.tensor(states), actions


@cache
def get_solved_state():
    fc = FaceCube()
    return torch.tensor(face_str2int(fc.to_string()))


def StateLoader(val_test_rate=0.1, batch_size=32, size=None):
    collate_fn = make_state_and_solve_state

    data = R222ShortestAll(size=size)
    train_data_start = 1 # ignore the first data because it is solved state
    train_data_end = int(len(data)*(1 - 2*val_test_rate))
    train_dataloader = DataLoader(data[train_data_start:train_data_end],
                                    batch_size=batch_size,
                                    shuffle=True,
                                    collate_fn=collate_fn)
    val_data_start = train_data_end
    val_data_end = train_data_end + int(len(data)*val_test_rate)
    val_dataloader = DataLoader(data[val_data_start:val_data_end],
                                batch_size=batch_size,
                                shuffle=False,
                                collate_fn=collate_fn)
    test_data_start = val_data_end
    test_data_end = val_data_end + int(len(data)*val_test_rate)
    test_dataloader = DataLoader(data[test_data_start:test_data_end],
                                    batch_size=batch_size,
                                    shuffle=False,
                                    collate_fn=collate_fn)
    return train_dataloader, val_dataloader, test_dataloader


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


def StateLoader2(val_test_rate=0.1, batch_size=32, size=None):
    def collate_fn(batch):
        start_state, solve_states = make_state_and_solve_state(batch)
        inputs = solve_states[:]
        inputs[:, 0] = start_state
        targets = solve_states[:, 1:]
        return inputs, targets

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


def AllLoader(val_test_rate=0.1, batch_size=32, size=None):
    data = R222ShortestAll(size=size)
    def collate_fn(batch):
        moves = get_moves(batch)
        start_state, solve_states = make_state_and_solve_state(batch)
        inputs = solve_states[:]
        inputs[:, 0] = start_state
        targets = solve_states[:, 1:]
        return inputs, [targets, moves]

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


def StateDistanceLoader(val_test_rate=0.1, batch_size=32, size=None):
    data = R222ShortestAll(size=size)
    def collate_fn(batch):
        state, distance = make_state_and_distance(batch)
        return state, distance

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


def StateNextActionLoader(val_test_rate=0.1, batch_size=32, size=None):
    data = R222ShortestAll(size=size)
    def collate_fn(batch):
        state, next_action = make_state_and_action(batch)
        return state, next_action

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


def StateActionLoader(val_test_rate=0.1, batch_size=32, size=None):
    data = R222ShortestAll(size=size)
    def collate_fn(batch):
        states, actions = make_state_action_sequence(batch)
        return states, actions

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
    else:
        raise ValueError("Invalid loder_name: {}".format(loder_name))


if __name__ == '__main__':
    data = R222ShortestAll()
    print("data size", len(data))
    data = R222ShortestAll(size=100000)
    print("data size", len(data))
    print("example data[1]", data[1])
    """
    train_dataloader = DataLoader(data[:3000000], batch_size=4, shuffle=True)
    for i in train_dataloader:
        print("batch_size", len(i))
        print(i[0])
        break

    """
    print("NOPLoader")
    train_dataloader, val_dataloader, test_dataloader = create_dataloader("NOPLoader", 0.1, 10)
    for i in train_dataloader:
        print("src, tgt", len(i))
        print("src.shape", i[0].shape)
        print("tgt.shape", i[1].shape)
        print("src[0]", i[0][0])
        print("src[0] max:", i[0][0].max())
        print("tgt[0]", i[1][0])
        break

    print("StateLoader")
    train_dataloader, val_dataloader, test_dataloader = create_dataloader("StateLoader", 0.1, 10)
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
        print("src[0]", i[0][0])
        print("src[0] max:", i[0][0].max())
        print("tgt[0]", i[1][0])
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