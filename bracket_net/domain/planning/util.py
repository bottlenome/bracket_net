from ...model import encoder

import pytorch_lightning as L
import torch
import torch.nn as nn
from neural_astar.planner.astar import VanillaAstar


def calc_accuracy(outputs, out_trajs, ignore_index):
    zero_mask = (out_trajs != 0)
    ignore_mask = (out_trajs != ignore_index)
    total = (ignore_mask*zero_mask).float().sum(dim=(1, 2)) + 0.1e-10
    same = ((outputs.argmax(dim=1) == out_trajs)*ignore_mask*zero_mask).float().sum(dim=(1, 2))
    return (same / total).mean()

def calc_entropy(outputs):
    """Calculate entropy of the outputs
        Args: outputs: [batch, n_vocab, seq]
    """
    outputs = nn.functional.softmax(outputs, dim=1)
    return -1 * (outputs * outputs.log()).sum(dim=1).mean()

class CommonModule(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.lr = config.params.lr
        self.vanilla_astar = VanillaAstar()
        self.is_1d = (config.model.type == "1d")
        self.is_2d = (config.model.type == "2d")
        self.problem_start = nn.Parameter(
            torch.zeros(config.params.batch_size, 1) + 2, requires_grad=False)
        self.estimate_start = nn.Parameter(
            torch.zeros(config.params.batch_size, 1) + 3, requires_grad=False)
        self.estimate_end = nn.Parameter(
            torch.zeros(config.params.batch_size, 2) + 4, requires_grad=False)
        self.ignore_index=5
        self.initial_step=4000
        self.enable_entropy_loss = config.params.enable_entropy_loss


    def forward(self, map_designs, start_maps, goal_maps, out_trajs=None):
        raise NotImplementedError

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(),
                                   self.lr)

    def loss(self, outputs, train_set, start_maps):
        if self.is_1d:
            loss = self.calc_1d_loss(outputs, train_set, start_maps)
        elif self.is_2d:
            loss =  self.calc_2d_loss(outputs, train_set, start_maps)
        else:
            loss = self.calc_map_loss(outputs, train_set)
        entropy = calc_entropy(outputs)
        assert(entropy >= 0)
        assert(entropy <= torch.log(torch.tensor(self.d_vocab, dtype=torch.float32)))
        if self.enable_entropy_loss:
            entropy_loss = torch.log(torch.tensor(self.d_vocab, dtype=torch.float32)) - entropy
            if self.initial_step >= self.global_step:
                l = 1.0
            elif self.initial_step * 10 >= self.global_step:
                l = (self.initial_step * 10 - self.global_step) / (self.initial_step * 10) * 0.9 + 0.001
            else:
                l = 0.001
            loss += l * entropy_loss
        else:
            entropy_loss = torch.tensor(0.0)
        return loss, entropy, entropy_loss


    def calc_1d_loss(self, outputs, out_trajs, start_maps):
        # [batch, n_vocab, seq] -> [batch, seq, n_vocab]
        outputs = outputs.permute(0, 2, 1)
        (batch, seq, n_vocab) = outputs.size()
        outputs = outputs.reshape(batch*seq, n_vocab)
        # 1 + 32*32 + 32*32 + 32*32 + 1 + 32*32 + 1 ->
        #  32*32 + 32*32 + 32*32 + 1 + 32*32 + 1
        ignore = (torch.zeros_like(out_trajs, dtype=torch.int64) + self.ignore_index)
        # batch, 32, 32 -> batch, 32*32
        ignore = ignore.view(ignore.size(0), -1)
        ignore2 = ignore.clone()
        ignore3 = ignore.clone()
        # batch, 1, 32, 32 -> batch, 32*32
        train_trajs = out_trajs.view(out_trajs.size(0), -1)
        ignore4 = ignore[:, :1].clone()
        train_set = torch.cat([ignore, ignore2, ignore3,
                               self.estimate_start,
                               train_trajs,
                               self.estimate_end,
                               ignore4], dim=1)
        train_set = train_set.to(outputs.device)
        train_set = train_set.to(torch.int64)
        train_set = train_set.view(-1)
        loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)(outputs, train_set)
        return loss

    def calc_2d_loss(self, outputs, out_trajs, start_maps):
        # 1 + 32*32 + 1 + 32*32 + 1 ->
        #  32*32 + 1 + 32*32 + 1
        ignore = (torch.zeros_like(start_maps, dtype=torch.int64) + self.ignore_index)
        # batch, 32, 32 -> batch, 32*32
        ignore = ignore.view(ignore.size(0), -1)
        # batch, 1, 32, 32 -> batch, 32*32
        train_trajs = out_trajs.view(out_trajs.size(0), -1)
        ignore2 = ignore[:, :1].clone()
        train_set = torch.cat([ignore, self.estimate_start,
                               train_trajs,
                               self.estimate_end,
                               ignore2], dim=1)
        train_set = train_set.to(outputs.device)
        train_set = train_set.to(torch.int64)
        loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)(outputs, train_set)
        return loss

    def calc_map_loss(self, outputs, out_trajs):
        out_trajs = out_trajs.view(out_trajs.size(0),
                                   out_trajs.size(2),
                                   out_trajs.size(3))
        out_trajs = out_trajs.to(torch.int64)
        loss = nn.CrossEntropyLoss()(outputs, out_trajs)
        return loss

    def training_step(self, train_batch, batch_idx):
        map_designs, start_maps, goal_maps, out_trajs = train_batch
        outputs = self.forward(map_designs, start_maps, goal_maps, out_trajs)
        loss, entropy, entropy_loss = self.loss(outputs, out_trajs, start_maps)
        self.log("metrics/train_loss", loss, prog_bar=True)
        self.log("metrics/entropy", entropy)
        return loss + entropy_loss

    def validation_step(self, val_batch, batch_idx):
        map_designs, start_maps, goal_maps, out_trajs = val_batch
        # outputs: [batch, n_vocal, seq]
        # seq: 1 + 32*32 + 32*32 + 32*32 + 1 + 32*32 + 2
        outputs = self.forward(map_designs, start_maps, goal_maps, out_trajs)
        if self.is_1d:
            loss = self.calc_1d_loss(outputs, out_trajs, start_maps)
            entropy = calc_entropy(outputs)
            outputs = outputs[:, :, 32*32*3+1:32*32*3+1+32*32]
            outputs = outputs.view(outputs.size(0), -1, 32, 32)
        elif self.is_2d:
            loss = self.calc_2d_loss(outputs, out_trajs, start_maps)
            outputs = outputs[:, :, 32*32+1:32*32+1+32*32]
            outputs = outputs.view(outputs.size(0), -1, 32, 32)
        else:
            loss = self.calc_map_loss(outputs, out_trajs)
        self.log("metrics/val_loss", loss, prog_bar=True)
        self.log("metrics/val_entropy", entropy)
        accu = calc_accuracy(outputs, out_trajs, self.ignore_index)
        self.log("metrics/val_accu", accu)
        path = outputs.argmax(dim=1)
        path = path.view(-1, 1, path.size(1), path.size(2))
        p_opt = get_p_opt(out_trajs,
                          map_designs, start_maps, goal_maps, path)
        self.log("metrics/p_opt", p_opt)

        self.log("metrics/p_exp", 0)
        self.log("metrics/h_mean", 0)
        if batch_idx == 0:
            import wandb
            img = outputs[0].detach().argmax(dim=0)
            img = img * 255.
            img = img.cpu().numpy()
            self.logger.experiment.log({
                "image/estimated_traj": wandb.Image(img)
            })
            img = out_trajs[0].detach()
            img = img * 255.
            img = img.cpu().numpy()
            self.logger.experiment.log({
                "image/true_traj": wandb.Image(img)
            })
        return loss

    def test_step(self, test_batch, batch_idx):
        map_designs, start_maps, goal_maps, out_trajs = test_batch
        outputs = self.forward(map_designs, start_maps, goal_maps, None)
        size = map_designs.size(2) * map_designs.size(3)
        # the outputs shift 1 so ignore first element
        index = 32*32 + 32*32 + 32*32 + 1
        estimated_traj = []
        # outputs becomes 1 + 32*32 * 3 + 1 + 32*32 + 2
        for i in range(size):
            # expected shape is batch, n_vocab, seq
            # get one element
            estimated_element = outputs[:, :, index+i:index+i+1]
            estimated_traj.append(estimated_element.detach().clone())
            del outputs
            torch.cuda.empty_cache()
            # [(batch, n_vocab, 1), ...] -> batch, n_vocab, seq
            answers = torch.stack(estimated_traj, dim=2)
            # batch, n_vocab, seq -> batch, seq
            answers = answers.argmax(dim=1)
            # batch, seq -> batch, 1, seq
            answers = answers.view(answers.size(0), 1, -1)
            outputs = self.forward(
                    map_designs, start_maps, goal_maps, answers)
        # [(batch, n_vocab, 1), ...] -> batch, n_vocab, seq
        answers = torch.stack(estimated_traj, dim=2).view(
                outputs.size(0), -1, size)
        outputs[:, :, size*3:size*3+size] = answers
        if self.is_1d:
            loss = self.calc_1d_loss(outputs, out_trajs, start_maps)
            outputs = outputs[:, :, index:index+32*32]
            outputs = outputs.view(outputs.size(0), -1, 32, 32)
        elif self.is_2d:
            loss = self.calc_2d_loss(outputs, out_trajs, start_maps)
            outputs = outputs[:, :, index:index+32*32]
            outputs = outputs.view(outputs.size(0), -1, 32, 32)
        else:
            loss = self.calc_map_loss(outputs, out_trajs)
        self.log("metrics/test_loss", loss, prog_bar=True)
        accu = calc_accuracy(outputs, out_trajs, self.ignore_index)
        self.log("metrics/test_accu", accu)
        path = outputs.argmax(dim=1)
        path = path.view(-1, 1, path.size(1), path.size(2))
        p_opt = get_p_opt(out_trajs,
                            map_designs, start_maps, goal_maps, path)
        self.log("metrics/test_p_opt", p_opt)
        if batch_idx == 0:
            import wandb
            img = outputs[0].detach().argmax(dim=0)
            img = img * 255.
            img = img.cpu().numpy()
            self.logger.experiment.log({
                "image/test_traj": wandb.Image(img)
            })
            img = out_trajs[0].detach()
            img = img * 255.
            img = img.cpu().numpy()
            self.logger.experiment.log({
                "image/test_true_traj": wandb.Image(img)
            })
        return loss


def get_p_opt(out_trajs, map_designs, start_maps, goal_maps, paths):
    if map_designs.shape[1] == 1:
        # va_outputs = vanilla_astar(map_designs, start_maps, goal_maps)
        # pathlen_astar = va_outputs.paths.sum(
        #                   (1, 2, 3)).detach().cpu().numpy()
        pathlen_astar = out_trajs.sum((1, 2, 3)).detach().cpu().numpy()
        pathlen_model = paths.sum((1, 2, 3)).detach().cpu().numpy()
        p_opt = (pathlen_astar == pathlen_model).mean()
        return p_opt


class NaiveBase(CommonModule):
    def __init__(self, config, Model):
        super().__init__(config)
        self.d_vocab = config.gpt.d_vocab
        self.d_model = config.gpt.d_model
        self.model = Model(self.d_vocab,
                           encoder.PostionalEncodingFactory(
                               "1d", max_len=32*32*4+4+1),
                           d_model=self.d_model,
                           n_head=config.gpt.n_head,
                           num_layers=config.gpt.num_layers,
                           dropout=config.gpt.dropout)

    def forward(self, map_designs, start_maps, goal_maps, out_trajs):
        # batch, 1, 32, 32 -> batch, 32 * 32
        start_maps = start_maps.view(start_maps.size(0), -1)
        goal_maps = goal_maps.view(goal_maps.size(0), -1)
        map_designs = map_designs.view(map_designs.size(0), -1)
        # concat problem_start, start_maps, goal_maps, map_designs,
        #        estimate_start, out_trajs, estimate_end
        if out_trajs is not None:
            src = torch.cat([self.problem_start,
                            start_maps, goal_maps, map_designs,
                            self.estimate_start,
                            out_trajs.view(out_trajs.size(0), -1),
                            self.estimate_end], dim=1)
        else:
            src = torch.cat([self.problem_start,
                            start_maps, goal_maps, map_designs,
                            self.estimate_start], dim=1)
        # float to int
        src = src.to(torch.int64)
        # batch, seq -> seq, batch
        src = src.permute(1, 0)
        # seq, batch -> seq, batch, d_vocab
        out = self.model(src)
        # seq, batch, d_vocab -> batch, d_vocab, seq
        out = out.permute(1, 2, 0)
        return out


class SeqEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=1)

    def forward(self, src):
        out = self.model(src)
        # batch, out_channels, seq -> seq, batch, out_channels
        out = out.permute(2, 0, 1)
        return out


class NNAstarLikeBase(CommonModule):
    def __init__(self, config, Model):
        super().__init__(config)
        d_vocab = config.gpt.d_vocab
        d_model = config.gpt.d_model
        self.d_vocab = d_vocab
        self.condition_encode = nn.Conv2d(in_channels=3,
                                          out_channels=1,
                                          kernel_size=1)
        self.seq_embedding = SeqEmbedding(1, d_model)

        self.model = Model(d_vocab,
                           encoder.PostionalEncodingFactory(
                               "none", height=32, width=32),
                           d_model=d_model,
                           n_head=config.gpt.n_head,
                           num_layers=config.gpt.num_layers,
                           dropout=config.gpt.dropout,
                           embed=self.seq_embedding)
        self.remap = nn.Softmax(dim=-1)

    def forward(self, map_designs, start_maps, goal_maps, out_trajs):
        # batch, 1, 32, 32 -> batch, 3, 32, 32
        src = torch.cat([map_designs, start_maps, goal_maps], dim=1)
        # batch, 3, 32, 32 -> batch, 1, 32, 32
        encoded = self.condition_encode(src)
        # batch, 1, 32, 32 -> batch, 32 * 32
        encoded = encoded.view(encoded.size(0), -1)
        # problem_start: batch, 1
        # concat problem_start, encoded,
        #        estimate_start, out_trajs, estimate_end
        src1 = torch.cat([self.problem_start, encoded,
                          self.estimate_start,
                          out_trajs.view(out_trajs.size(0), -1),
                          self.estimate_end], dim=1)
        # batch, seq -> batch, 1, seq
        src1 = src1.view(src1.size(0), 1, -1)
        # batch, 1, seq -> seq, batch, d_vocab
        out = self.model(src1)
        # seq, batch, d_vocab -> seq, batch, d_vocab
        out = self.remap(out)
        # seq, batch, d_vocab -> batch, d_vocab, seq
        out = out.permute(1, 2, 0)
        return out

if __name__ == "__main__":
    import math

    out_trajs = torch.zeros(2, 32, 32)
    out_trajs[0, 0] = 1
    out_trajs[1, 1:3] = 1
    outputs = torch.zeros(2, 6, 32, 32)
    outputs[:, 0, :, :] = 1
    accu = calc_accuracy(outputs, out_trajs, 5)
    assert(accu == 0.0)
    print(accu.item())
    outputs[:, 0, :, :] = 0
    outputs[0, 1, 0, 0] = 1
    accu = calc_accuracy(outputs, out_trajs, 5)
    assert(math.fabs(accu.item() - 1./32/2) < 0.00001)
    print(accu.item())
    outputs[0, 1, 0] = 1
    accu = calc_accuracy(outputs, out_trajs, 5)
    assert(accu == 0.5)
    print(accu.item())
