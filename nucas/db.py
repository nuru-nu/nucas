import glob
import json
import logging
import os

import mediapy
import ml_collections
import pandas as pd
import torch
import tqdm

from . import utils

# TODO: refactor torch code


def flatten(d, sep='.', prefix=()):
  for k, v in d.items():
    if isinstance(v, dict):
      yield from flatten(v, sep=sep, prefix=prefix + (k,))
    else:
      yield ('.'.join(prefix + (k,)), v)


class Db:

  def __init__(self, base, limit=None, reload=False):
    self.base = base
    self.ids = [
        path.split('/')[-1].split('.')[0]
        for path in sorted(glob.glob(self.path('*.json')))
        if not path.endswith('_metrics.json')
    ]
    if limit:
      self.ids = self.ids[:limit]
    self._csv_path = self.path('_db.csv')
    # TODO: merge stats from insights Colab
    if os.path.exists(self._csv_path) and not reload:
      self.df = pd.read_csv(self._csv_path)
    else:
      self.df = self._df_load()
      self._df_save()

  def path(self, name):
    return os.path.join(self.base, name)

  def save_video(self, config, imgs, *, overwrite=False):
    if len(self.df):
      assert config.id not in set(self.df.id) or overwrite

    p = lambda ext: self.path(f'{config.id}.{ext}')

    mediapy.write_video(p('mp4'), imgs)
    mediapy.write_image(p('png'), imgs[-1])
    mediapy.write_image(p('jpg'), imgs[-1], fmt='jpeg')

  def save(self, config, model, stats, imgs, *, overwrite=False):
    if len(self.df):
      assert config.id not in set(self.df.id) or overwrite

    p = lambda ext: self.path(f'{config.id}.{ext}')

    self.save_video(config, imgs, overwrite=overwrite)

    torch.save(model, p('pt'))
    with open(p('stats'), 'w') as f:
      json.dump(stats, f)
    with open(p('json'), 'w') as f:
      json.dump(config.to_dict(), f, indent=2)

    self.df = pd.concat(
        [
            self.df.query(f'id!="{config.id}"') if len(self.df) else self.df,
            pd.DataFrame(
                [{**dict(flatten(config.to_dict())), **dict(flatten(stats))}]
            ),
        ]
    )
    self._df_save()

  def load(self, id_):
    """Returns config, model, stats for the given id."""
    with open(self.path(f'{id_}.json')) as f:
      config = ml_collections.ConfigDict(json.load(f))
    # model = globals()[config.model_name](**config.model)
    model = torch.load(self.path(f'{id_}.pt'))
    with open(self.path(f'{id_}.stats')) as f:
      stats = json.load(f)
    return config, model, stats

  def _df_save(self):
    with open(self._csv_path, 'w') as f:
      self.df.to_csv(f, index=False)

  def _df_load(self):
    _config = lambda id_: json.load(open(self.path(f'{id_}.json')))
    _stats = lambda id_: {
        k: v
        for k, v in json.load(open(self.path(f'{id_}.stats'))).items()
        if k not in ('loss_log',)
    }
    return pd.DataFrame(
        {**dict(flatten(_config(id_))), **dict(flatten(_stats(id_)))}
        for id_ in tqdm.tqdm(self.ids)
    )


unique = lambda df: df[[c for c in df.columns if len(df[c].unique()) > 1]]


def get_db(base=None):
  if base is None:
    base = utils.get_basedir()
  db = Db(base)
  logging.info('Loaded db from %s: %d entries', base, len(db.df))
  return db
