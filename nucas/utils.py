import datetime
import io
import os
import logging

import numpy as np
import PIL.Image
import requests
import torch

# TODO: refactor pytorch code

# TODO: utilities for creating npz files


_basedir = f'{os.path.expanduser("~")}/ncas'


def get_basedir():
  if not os.path.exists(_basedir):
    os.makedirs(_basedir)
    logging.info('Created basedir: %s', _basedir)
  return _basedir


def set_basedir(basedir):
  global _basedir
  _basedir = basedir
  logging.info('Updated basedir: %s', basedir)


def ts():
  return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


def imread(url, mode='RGB'):
  if url.startswith(('http:', 'https:')):
    # wikimedia requires a user agent
    headers = {
        'User-Agent': 'Requests in Colab/0.0 (https://colab.research.google.com/; no-reply@google.com) requests/0.0'
    }
    r = requests.get(url, headers=headers)
    f = io.BytesIO(r.content)
  else:
    f = url
  im = PIL.Image.open(f)
  if mode is not None:
    im = im.convert(mode)
  return im


def im2np(im):
  a = np.array(im)[..., :3] / 255
  assert a.ndim == 3
  if a.shape[-1] == 1:
    a = np.tile(a, (1, 1, 3))
  return a.astype('float32')


def imth(im, sz):
  im = im.copy()
  im.thumbnail((sz, sz), PIL.Image.ANTIALIAS)
  return im


def np2im(a):
  return PIL.Image.fromarray((255 * a).astype('uint8'))


def np2pt(a):
  t = torch.as_tensor(a)
  if len(t.shape) == 3:
    t = t[None, ...]
  return t.permute(0, 3, 1, 2)


def pt2np(t):
  return t.permute([0, 2, 3, 1]).cpu().numpy()


def im2pt(im):
  return np2pt(im2np(im))


def perchannel_conv(x, filters):
  """filters: [filter_n, h, w]"""
  b, ch, h, w = x.shape
  y = x.reshape(b * ch, 1, h, w)
  y = torch.nn.functional.pad(y, [1, 1, 1, 1], 'circular')
  # [b*ch, 1, h, w]
  y = torch.nn.functional.conv2d(y, filters[:, None])
  # [b*ch, 4, h, w]
  return y.reshape(b, -1, h, w)
  # [b, ch*4, h, w]


def perception(x):
  ident = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
  sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
  lap = torch.tensor([[1.0, 2.0, 1.0], [2.0, -12, 2.0], [1.0, 2.0, 1.0]])
  filters = torch.stack([ident, sobel_x, sobel_x.T, lap])
  return perchannel_conv(x, filters)


def to_rgb(x):
  return x[..., :3, :, :] + 0.5
