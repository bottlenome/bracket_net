from __future__ import annotations

import hydra
import pytorch_lightning as pl
import torch

from bracket_net.data.breakout import create_dataloader

import gym
import moviepy.editor as mpy
from gym.wrappers import AtariPreprocessing
import numpy as np
import collections


def evaluate(env, model, context_length,
             target_rtg: float = 70.0, filename="replay.gif"):
    imgs = []
    frames = collections.deque(maxlen=4)
    for _ in range(4):
        frames.append(np.zeros((84, 84), dtype=np.float32))

    frame = env.reset()
    frames.append(frame)

    rtgs = collections.deque([], maxlen=context_length)
    states = collections.deque([], maxlen=context_length)
    actions = collections.deque([], maxlen=context_length-1)
    timesteps = collections.deque([], maxlen=context_length)

    done = False
    sum_rewards = 0
    score = 0

    for step in range(1600):
        timesteps.append(step)
        rtgs.append(max(target_rtg - sum_rewards, 0))
        states.append(np.stack(frames, axis=2).astype(np.float32))

        action = model.act(rtgs, states, actions, timesteps[0])

        next_state, reward, done, _ = env.step(action)
        frames.append(next_state)
        actions.append(action)
        score += reward
        sum_rewards += np.clip(reward, 0., 1.)

        if done:
            break

    print(f"score: {score}, sum_rewards:{sum_rewards}, step: {step}")


def create_gif(env, model, filename='replay.gif'):
    frames = []
    state = env.reset()
    done = False
    score = 0

    while not done:
        frame = env.render(mode='rgb_array')
        frames.append(frame)
        action = model.act(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        score += reward
    print(f"score: {score}")
    clip = mpy.ImageSequenceClip(frames, fps=30)
    clip.write_gif(filename, fps=30)


@hydra.main(config_path="config", config_name="train_breakout")
def main(config):
    if config.model.name == "up-causal-naive":
        from bracket_net.domain.breakout.up_causal_unet import DecisionLike
        model = DecisionLike(config)
    else:
        raise ValueError(f"Unknown model name {config.model.name}")
    train_loader, val_loader, test_loader = create_dataloader(
                                                 config.data.name,
                                                 config.data.val_test_rate,
                                                 config.params.batch_size,
                                                 num_steps=100000,
                                                 context_length=30)
    profiler = None
    wandb_logger = pl.loggers.WandbLogger(
            project=config.project, name=config.model.name, log_model=True)
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=1,
        logger=wandb_logger,
        max_epochs=config.params.num_epochs,
        profiler=profiler
    )
    trainer.fit(model, train_loader, val_loader)

    model.eval()
    env = gym.make("BreakoutNoFrameskip-v4")
    env = AtariPreprocessing(env, screen_size=84, frame_skip=4)
    evaluate(env, model, 10)
    env.close()


if __name__ == "__main__":
    main()
