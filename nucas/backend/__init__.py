backend = None
try:
  import torch as unused_torch

  backend = 'torch'
except ImportError:
  backend = 'tensorflow'

print(f'Using backend: {backend}')

if backend == 'torch':
  from . import torch as backend

elif backend == 'tensorflow':
  from . import tensorflow as backend

else:
  raise ImportError(f'Unsupported backend: {backend}')

CaOrig = backend.CaOrig
MuOrig = backend.MuOrig
load = backend.load
profile = backend.profile
