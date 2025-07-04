import logging

backend = None
try:
  import torch as unused_torch

  backend = 'torch'
except ImportError:
  backend = 'tensorflow'

logging.info(f'Using backend: {backend}')

if backend == 'torch':
  from . import torch as backend

elif backend == 'tensorflow':
  from . import tensorflow as backend

else:
  raise ImportError(f'Unsupported backend: {backend}')

CaOrig = backend.CaOrig
CaOt = backend.CaOt
MuOrig = backend.MuOrig
MuOt = backend.MuOt
load = backend.load
profile = backend.profile
