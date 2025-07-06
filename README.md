# nucas - nuru NCAs

Library to be used on Colab or for local training of NCAs. Based on a collection
of Colabs mentioned in https://nuru.nu/nca-doc

Don't know NCAs? Check out
https://distill.pub/selforg/2021/textures/

NCA code based on
[google-research/self-organizing-systems](https://github.com/google-research/self-organising-systems).

TODO: should we use github issues for TODOs ?

TODO: write documentation (pydoc / examples)

## Synopsis

### Colab

Demonstration Colab:
https://colab.research.google.com/github/nuru-nu/nucas/blob/main/notebooks/train_run.ipynb

Single install should work:

```jupyter
!pip install -q git+https://github.com/nuru-nu/nucas#egg=nucas[colab]
```

You probably want to run below code on a runtime with GPUs:

```python
import nucas
nucas.notebook.init()  # will ask to authorize Drive access ...

config = nucas.train.get_config()
config.target = 'https://www.robots.ox.ac.uk/~vgg/data/dtd/images/chequered/chequered_0045.jpg'

model, stats = nucas.train.train(config)
imgs = nucas.run.run(config, model)

import mediapy
mediapy.show_video(imgs)

db = nucas.db.get_db()
db.save(config, model, stats, imgs, overwrite=True)
```

### Run on OS X

Below instructions were tested on OS X 15.5

1. Install the repo: `git clone git@github.com:nuru-nu/nucas.git`

2. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`

3. Install virtual environment: `uv venv --python 3.13 && . .venv/bin/activate`

4. Install dependencies: `uv pip install -e '.[pytorch,dev,jupyter]'`

NOTE: If `torch` fails with some `_lzma` error, make sure to install xz via
brew: `brew install xz`.

TODO: do we want to add UI ? browser based ?

TODO: create OS X application ? a Docker container ?

```python
import nucas

config = nucas.train.get_config()
config.target = 'https://www.robots.ox.ac.uk/~vgg/data/dtd/images/chequered/chequered_0045.jpg'

model, stats = nucas.train.train(config)
imgs = nucas.run.run(config, model)

import mediapy
mediapy.show_video(imgs)

db = nucas.db.get_db()
db.save(config, model, stats, imgs)
```

### Run in Browser

TODO: convert ncas for ca.js

TODO: reimplement ca.js with swiss.gl

TODO: build server for midi input / syphon output
