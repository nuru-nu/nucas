import pathlib
import tempfile

import PIL.Image
import torch

import nucas


def get_tmpimg(tmpdir):
  img = PIL.Image.new('RGB', (1, 1), color='black')
  img_path = pathlib.Path(tmpdir) / 'black_pixel.jpg'
  img.save(img_path, format='JPEG')
  return img_path.as_posix()


def get_test_config(target):
  config = nucas.train.get_config()
  config.target = target
  config.steps = 2
  config.sz = 32
  config.rollout_min = 2
  config.rollout_max = 4
  config.update_rate = 0.5
  config.pool_size = 8
  config.batch_size = 2
  return config


def test_train_run():
  with tempfile.TemporaryDirectory() as tmpdir:
    config = get_test_config(get_tmpimg(tmpdir))
    model, stats = nucas.train.train(config)
    assert list(stats) == ['dt', 'loss_log', 'params', 'flops']
    imgs = nucas.run.run(config, model, steps=2, sz=4)
    assert imgs.shape == (2, 4, 4, 3)


def test_train_parent():
  with tempfile.TemporaryDirectory() as tmpdir:
    nucas.utils.set_basedir(tmpdir)
    config = get_test_config(get_tmpimg(tmpdir))
    model, _ = nucas.train.train(config)
    tmpmodel = (pathlib.Path(tmpdir) / 'parent.pt').as_posix()
    torch.save(model, tmpmodel)
    config.parent = 'parent'
    _, _ = nucas.train.train(config)
