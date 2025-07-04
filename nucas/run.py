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
    x = model.seed(1, sz)
    for _ in tqdm.trange(steps, leave=False):
      for _ in range(per_step):
        x[:] = model(x, update_rate=update_rate)
      img = utils.pt2np(utils.to_rgb(x))[0]
      imgs.append(img)
  return np.array(imgs)


def run_graft(
    config,
    model_child,
    model_parent,
    steps=300,
    cycles=1,
    update_rate=None,
    *,
    mask_f,
    update_rate_f,
    per_step_f,
    sz=256,
):
  if update_rate is None:
    update_rate = config.update_rate

  imgs = []
  t = 0
  with torch.no_grad():
    x = model_child.seed(1, sz)

    for _ in range(cycles):
      for i in tqdm.trange(steps, leave=False):
        # Explicitly set dtype to float32 to be compatible with MPS
        mask = torch.tensor(mask_f(i / steps), dtype=torch.float32)

        update_rate, per_step = map(
            float, (update_rate_f(i / steps), per_step_f(i / steps))
        )
        while t < per_step:
          dx1 = model_child(x, update_rate=update_rate) - x
          dx2 = model_parent(x, update_rate=update_rate) - x
          x[:] = x + (1 - mask) * dx1 + mask * dx2
          t += 1
        t -= per_step
        img = utils.pt2np(utils.to_rgb(x))[0]
        imgs.append(img)
  return np.array(imgs)
