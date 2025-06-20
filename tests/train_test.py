import tempfile
import pathlib
import PIL.Image

import nucas


def test_train_run():
  with tempfile.TemporaryDirectory() as tmpdir:
    img = PIL.Image.new('RGB', (1, 1), color='black')
    img_path = pathlib.Path(tmpdir) / 'black_pixel.jpg'
    img.save(img_path, format='JPEG')

    config = nucas.train.get_config()
    config.target = img_path.as_posix()
    config.steps = 2
    config.sz = 32
    config.rollout_min = 2
    config.rollout_max = 4
    config.update_rate = 0.5
    config.pool_size = 8
    config.batch_size = 2

    model, stats = nucas.train.train(config)
    assert stats
    imgs = nucas.run.run(config, model, steps=2, sz=4)
    assert imgs.shape == (2, 4, 4, 3)
