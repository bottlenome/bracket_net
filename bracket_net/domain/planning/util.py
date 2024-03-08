import pytorch_lightning as L
import torch
import torch.nn as nn
from neural_astar.planner.astar import VanillaAstar

class CommonModule(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.lr = config.params.lr
        self.vanilla_astar = VanillaAstar()
        self.is_gpt = (config.model.name.find("gpt") != -1)
        self.problem_start = nn.Parameter(
            torch.zeros(config.params.batch_size, 1) + 2, requires_grad=False)
        self.estimate_start = nn.Parameter(
            torch.zeros(config.params.batch_size, 1) + 3, requires_grad=False)
        self.estimate_end = nn.Parameter(
            torch.zeros(config.params.batch_size, 2) + 4, requires_grad=False)
    
    def forward(self, map_designs, start_maps, goal_maps, out_trajs=None):
        raise NotImplementedError

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.model.parameters(),
                                   self.lr)
    
    def training_step(self, train_batch, batch_idx):
        map_designs, start_maps, goal_maps, out_trajs = train_batch
        outputs = self.forward(map_designs, start_maps, goal_maps, out_trajs)
        if self.is_gpt:
            # 1 + 32*32 + 32*32 + 32*32 + 1 + 32*32 + 1 ->
            #  32*32 + 32*32 + 32*32 + 1 + 32*32 + 1
            ignore = (torch.zeros_like(start_maps, dtype=torch.int64) - 100)
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
            loss = nn.CrossEntropyLoss()(outputs, train_set)
        else:
            out_trajs = out_trajs.view(out_trajs.size(0),
                                    out_trajs.size(2),
                                    out_trajs.size(3))
            out_trajs = out_trajs.to(torch.int64)
            loss = nn.CrossEntropyLoss()(outputs, out_trajs)
        self.log("metrics/train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        map_designs, start_maps, goal_maps, out_trajs = val_batch
        outputs = self.forward(map_designs, start_maps, goal_maps, out_trajs)
        if self.is_gpt:
            # 1 + 32*32 + 32*32 + 32*32 + 1 + 32*32 + 2 ->
            #  32*32 + 32*32 + 32*32 + 1 + 32*32 + 1
            ignore = torch.zeros_like(start_maps, dtype=torch.int64) - 100
            # batch, 32, 32 -> batch, 32*32
            ignore = ignore.view(ignore.size(0), -1)
            ignore2 = ignore.clone()
            ignore3 = ignore.clone()
            # batch, 1, 32, 32 -> batch, 32*32
            train_trajs = out_trajs.view(out_trajs.size(0), -1)
            ignore4 = ignore[:, :1].clone()
            val_set = torch.cat([ignore, ignore2, ignore3,
                                 self.estimate_start,
                                 train_trajs,
                                 self.estimate_end,
                                 ignore4], dim=1)
            val_set = val_set.to(outputs.device)
            val_set = val_set.to(torch.int64)
            loss = nn.CrossEntropyLoss()(outputs, val_set)
            outputs = outputs[:, :, 32*32*3+1:32*32*3+1+32*32]
            outputs = outputs.view(outputs.size(0), -1, 32, 32)
        else:
            out_trajs = out_trajs.view(out_trajs.size(0),
                                       out_trajs.size(2),
                                       out_trajs.size(3))
            out_trajs = out_trajs.to(torch.int64)
            loss = nn.CrossEntropyLoss()(outputs, out_trajs)
        self.log("metrics/val_loss", loss, prog_bar=True)
        accu = (outputs.argmax(dim=1) == out_trajs).float().mean()
        self.log("metrics/val_accu", accu)
        path = outputs.argmax(dim=1)
        path = path.view(-1, 1, path.size(1), path.size(2))
        p_opt = get_p_opt(self.vanilla_astar,
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


def get_p_opt(vanilla_astar, map_designs, start_maps, goal_maps, paths):
    if map_designs.shape[1] == 1:
        va_outputs = vanilla_astar(map_designs, start_maps, goal_maps)
        pathlen_astar = va_outputs.paths.sum((1, 2, 3)).detach().cpu().numpy()
        pathlen_model = paths.sum((1, 2, 3)).detach().cpu().numpy()
        p_opt = (pathlen_astar == pathlen_model).mean()
        return p_opt