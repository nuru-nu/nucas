import copy
import logging

import torch
import torch.nn.functional as F
import torchvision.models

from .. import utils


if torch.cuda.is_available():
  logging.info('CUDA available, using GPU')
  torch.set_default_device('cuda')
elif torch.backends.mps.is_available():
  logging.info('Apple Metal (MPS) available, using GPU')
  torch.set_default_device('mps')
else:
  logging.info('No GPU detected, using CPU')
  torch.set_default_device('cpu')

vgg16 = torchvision.models.vgg16(weights='IMAGENET1K_V1').features


def load(model, basename):
  model = copy.deepcopy(model)
  parent = torch.load(f'{basename}.pt')
  model.load_state_dict(parent.state_dict())
  return model


def get_grams(imgs):
  style_layers = [1, 6, 11, 18, 25]
  mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
  std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
  x = (imgs - mean) / std
  grams = []
  for i, layer in enumerate(vgg16[: max(style_layers) + 1]):
    x = layer(x)
    if i in style_layers:
      h, w = x.shape[-2:]
      y = x.clone()  # workaround for pytorch in-place modification bug(?)
      gram = torch.einsum('bchw, bdhw -> bcd', y, y) / (h * w)
      grams.append(gram)
  return grams


class CaOrig(torch.nn.Module):
  """See http://arxiv.org/abs/2105.07299"""

  def __init__(self, chn=12, hidden_n=96, bias=True, clip=None):
    super().__init__()
    self.chn = chn
    self.w1 = torch.nn.Conv2d(chn * 4, hidden_n, 1, bias=bias)
    self.w2 = torch.nn.Conv2d(hidden_n, chn, 1, bias=bias)
    self.w2.weight.data.zero_()
    self.clip = (
        (lambda x: torch.clip(x, -clip, clip)) if clip else (lambda x: x)
    )

  def forward(self, x, update_rate=0.5):
    y = utils.perception(x).contiguous()
    y = self.w2(torch.relu(self.w1(y)))
    b, c, h, w = y.shape
    udpate_mask = (torch.rand(b, 1, h, w) + update_rate).floor()
    return self.clip(x + y * udpate_mask)

  def seed(self, n, sz=128):
    return torch.zeros(n, self.chn, sz, sz)

  @classmethod
  def get_loss_f(cls, target):
    with torch.no_grad():
      grams_target = get_grams(target)

    def style_loss(imgs):
      loss = 0.0
      grams_imgs = get_grams(imgs)
      for x, y in zip(grams_imgs, grams_target):
        loss = loss + (x - y).square().mean()
      return loss

    return style_loss


def project_sort(x, proj):
  return torch.einsum('bcn,cp->bpn', x, proj).sort()[0]


def ot_loss(source, target, proj_n=32):
  ch, n = source.shape[-2:]
  projs = F.normalize(torch.randn(ch, proj_n), dim=0)
  source_proj = project_sort(source, projs)
  target_proj = project_sort(target, projs)
  target_interp = F.interpolate(target_proj, n, mode='nearest')
  return (source_proj - target_interp).square().sum()


def get_vgg_ot(imgs):
  style_layers = [1, 6, 11, 18, 25]
  mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
  std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
  x = (imgs - mean) / std
  b, c, h, w = x.shape
  features = [x.reshape(b, c, h * w)]
  for i, layer in enumerate(vgg16[: max(style_layers) + 1]):
    x = layer(x)
    if i in style_layers:
      b, c, h, w = x.shape
      features.append(x.reshape(b, c, h * w))
  return features


class CaOt(CaOrig):
  """See https://ieeexplore.ieee.org/document/9578591/"""

  @classmethod
  def get_loss_f(cls, target):
    with torch.no_grad():
      yy = get_vgg_ot(target)

    def style_loss(imgs):
      xx = get_vgg_ot(imgs)
      return sum(ot_loss(x, y) for x, y in zip(xx, yy))

    return style_loss


class _Mu(torch.nn.Module):

  def __init__(self, chn=12, filters=3):
    super().__init__()
    self.chn = chn
    ident = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    lap = torch.tensor([[1.0, 2.0, 1.0], [2.0, -12, 2.0], [1.0, 2.0, 1.0]])
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    )
    n = filters // 3

    filter_data = torch.stack(
        [ident]
        + [lap] * (n + (filters % 3 > 0))
        + [sobel_x] * (n + (filters % 3 > 1))
        + [sobel_x.T] * n
    )
    self.register_buffer('filters', filter_data)

    self.w = torch.nn.Conv2d(chn * (1 + filters) * 2, chn, 1, bias=True)
    self.w.weight.data.zero_()

  def forward(self, x, update_rate=0.5):
    p = utils.perchannel_conv(x, self.filters).contiguous()
    y = self.w(torch.concat([p, torch.abs(p)], dim=1))
    b, c, h, w = y.shape
    update_mask = (torch.rand(b, 1, h, w) + update_rate).floor()
    return x + update_mask * y

  def seed(self, n, sz=128, seed=0):
    # return torch.zeros(n, self.chn, sz, sz)
    torch.manual_seed(seed)
    return torch.randn(n, self.chn, sz, sz)  # * 1e-6


class MuOrig(_Mu):
  """See http://arxiv.org/abs/2111.13545"""

  @classmethod
  def get_loss_f(cls, target):
    return CaOrig.get_loss_f(target)


class MuOt(_Mu):
  """See http://arxiv.org/abs/2111.13545"""

  @classmethod
  def get_loss_f(cls, target):
    return CaOt.get_loss_f(target)


def profile(model):
  try:
    import thop
  except ImportError:
    logging.info('Could not import thop, skipping profiling')
    return 0, 0

  model_cpu = copy.deepcopy(model).to('cpu')

  seed_inputs = model_cpu.seed(1)
  if isinstance(seed_inputs, torch.Tensor):
    inputs_cpu = (seed_inputs.to('cpu'),)
  else:
    inputs_cpu = tuple(t.to('cpu') for t in seed_inputs)

  return thop.profile(model_cpu, inputs=inputs_cpu, verbose=False)


torch.serialization.add_safe_globals(
    [torch.nn.Conv2d, CaOrig, CaOt, MuOrig, MuOt]
)
