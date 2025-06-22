import numpy as np
import torch
import tqdm

from . import utils

# TODO: refactor pytorch code


def run(config, model, steps=300, per_step=1, update_rate=None, sz=256):
  if update_rate is None:
    update_rate = config.update_rate
  imgs = []
  with torch.no_grad():
    device = next(model.parameters()).device
    x = model.seed(1, sz).to(device)
    for _ in tqdm.trange(steps, leave=False):
      for _ in range(per_step):
        x[:] = model(x, update_rate=update_rate)
      img = utils.pt2np(utils.to_rgb(x))[0]
      imgs.append(img)
  return np.array(imgs)
