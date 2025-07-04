import datetime
import io
import os
import logging
import shutil
import subprocess
import tempfile

import numpy as np
import PIL.Image
import requests
import torch
import scipy.interpolate

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


def fetch(url):
  """Fetches with requests, fallback wget if available."""
  headers = {
      'User-Agent': 'Requests in Colab/0.0 (https://colab.research.google.com/; no-reply@google.com) requests/0.0'
  }
  r = requests.get(url, headers=headers)
  if r.status_code == 200:
    return r.content
  if shutil.which('wget') is None:
    raise RuntimeError(f'requests failed {r.status_code} - {r.text}')
  logging.info('requests failed, trying wget')
  with tempfile.TemporaryDirectory() as tmpdir:
    filename = f'{tmpdir}/img.jpg'
    result = subprocess.run(
        ['wget', '-O', filename, url], capture_output=True, text=True
    )
    if result.returncode != 0:
      raise RuntimeError(
          f'wget failed code={result.returncode}\n\n{result.stderr}\n\n{result.stdout}'
      )
    return open(filename, 'rb').read()


def imread(path_or_url, mode='RGB'):
  if path_or_url.startswith(('http:', 'https:')):
    f = io.BytesIO(fetch(path_or_url))
  else:
    f = path_or_url
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


def make_f(f, a):
  a = np.clip(a, 0, 1)
  return lambda x: f(
      scipy.interpolate.interp1d(np.linspace(0, 1, len(a)), a)(x)
  )
