import time

import ml_collections
import numpy as np
import torch

from . import backend
from . import notebook
from . import utils


def get_config():
  # TODO: replace with dataclass with pydoc
  config = ml_collections.ConfigDict()
  config.id = ''  # will be set in train()
  config.target = ''
  config.name = ''
  config.notes = ''
  config.group = ''
  config.sz = 128
  config.crop = 1.0
  config.crop_region = None  # use `RegionSelect()`
  config.lr = 1e-3
  config.steps = 1000
  config.model_name = 'CaOrig'
  config.model = {}
  config.parent = ''
  config.rollout_min = 32
  config.rollout_max = 64
  config.update_rate = 0.5
  config.pool_size = 1024
  config.batch_size = 4
  config.overflow_loss = True
  return config  # .lock()


def get_target(config):
  im = utils.imread(config.target)
  if config.crop_region:
    im = im.crop(config.crop_region)
  im = im.resize([config.sz, config.sz])
  return im.crop(
      (0, 0, int(config.crop * im.size[0]), int(config.crop * im.size[1]))
  )


def train(config, plot_every_n=4, iag=None):

  config.id = utils.ts()  # 🙈

  im_target = get_target(config)
  if iag is None:
    iag = notebook.ImageAndGraph(height=400, width=800)

  cls = getattr(backend, config.model_name)
  loss_f = cls.get_loss_f(utils.im2pt(im_target))
  model = cls(**config.model)

  if config.get('parent'):
    model = backend.load(model, f'{utils.get_basedir()}/{config.get("parent")}')

  # TODO: refactor to enable multiple backends

  model = model

  opt = torch.optim.Adam(model.parameters(), config['lr'])
  # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
  lr_sched = torch.optim.lr_scheduler.MultiStepLR(opt, [2000], 0.3)
  loss_log = []
  with torch.no_grad():
    pool = model.seed(config.pool_size, sz=config.sz)

  t0 = time.monotonic()
  for i in notebook.tqdm.trange(config.steps):
    with torch.no_grad():
      batch_idx = np.random.choice(len(pool), config.batch_size, replace=False)
      x = pool[batch_idx]
      if i % 8 == 0:
        # reinit 1/batch_size every 8th step
        x[:1] = model.seed(1, sz=config.sz)

    # The rollout loop now runs entirely on the GPU.
    step_n = np.random.randint(config.rollout_min, config.rollout_max)
    for k in range(step_n):
      x = model(x, update_rate=config.update_rate)

    loss = loss_f(utils.to_rgb(x))
    if not torch.isfinite(loss):
      print('infinite loss:', loss, x.mean())
    if config.overflow_loss:
      loss += (x - x.clamp(-1.0, 1.0)).abs().sum()

    with torch.no_grad():
      loss.backward()
      for p in model.parameters():
        p.grad /= p.grad.norm() + 1e-8
      opt.step()
      opt.zero_grad()
      lr_sched.step()
      pool[batch_idx] = x

      loss_log.append(loss.item())
      if i % plot_every_n == 0:
        iag.set_graph(range(len(loss_log)), loss_log)
        iag.set_image(
            np.hstack(
                [np.array(im_target) / 255.0]
                + list(utils.pt2np(utils.to_rgb(x)))
            )
        )

  dt = time.monotonic() - t0
  print(f'{dt:.1f} seconds - {len(loss_log) / dt:.1f} steps/sec')
  try:
    flops, params = backend.profile(model)
  except Exception as e:
    flops, params = 0, 0  # Assign default values on failure

  stats = dict(
      dt=dt,
      loss_log=loss_log,
      params=int(params),
      flops=int(flops),
  )

  # TODO: implement cache for Colab
  # TRAIN_CACHE[config.id] = (config, model, stats)
  # print(f'config, model, stats = TRAIN_CACHE[{config.id!r}]')

  return model, stats
