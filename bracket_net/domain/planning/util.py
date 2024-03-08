import pytorch_lightning as L
import torch
import torch.nn as nn
from neural_astar.planner.astar import VanillaAstar

class CommonModule(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.lr = config.params.lr
        self.vanilla_astar = VanillaAstar()
    
    def forward(self, map_designs, start_maps, goal_maps):
        raise NotImplementedError

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.model.parameters(),
                                   self.lr)
    
    def training_step(self, train_batch, batch_idx):
        map_designs, start_maps, goal_maps, out_trajs = train_batch
        outputs = self.forward(map_designs, start_maps, goal_maps)
        out_trajs = out_trajs.view(out_trajs.size(0),
                                   out_trajs.size(2),
                                   out_trajs.size(3))
        out_trajs = out_trajs.to(torch.int64)
        loss = nn.CrossEntropyLoss()(outputs, out_trajs)
        self.log("metrics/train_loss", loss)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        map_designs, start_maps, goal_maps, out_trajs = val_batch
        outputs = self.forward(map_designs, start_maps, goal_maps)
        out_trajs = out_trajs.view(out_trajs.size(0),
                                   out_trajs.size(2),
                                   out_trajs.size(3))
        out_trajs = out_trajs.to(torch.int64)
        loss = nn.CrossEntropyLoss()(outputs, out_trajs)
        self.log("metrics/val_loss", loss)
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