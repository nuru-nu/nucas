# nucas - nuru NCAs

Library to be used on Colab or for local training of NCAs. Based on a collection
of Colabs mentioned in https://nuru.nu/nca-doc

TODO: should we use github issues for TODOs ?

TODO: write documentation (pydoc / examples)

## Synopsis

### Colab

Demonstration Colab:
https://colab.research.google.com/drive/1f3NvI5GvFhS2j-Qo6STSAzofuldnFu0F

Single install should work:

```jupyter
!pip install -q git+https://github.com/nuru-nu/nucas#egg=nucas[colab]
```

You probably want to run below code on a runtime with GPUs:

```python
import nucas
nucas.colab.init()  # will ask to authorize Drive access ...

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

Unfortunately, default OS X installation does not include package `lzma`, so we
first need to install a working Python version:

```bash
brew install pyenv xz &&
eval "$(pyenv init -)" &&
pyenv update &&
(pyenv uninstall 3.10.17 || true) &&
pyenv install 3.10.17 &&
pyenv local 3.10.17
```

Then all dependencies can be installed (in developer mode) via `.[pytorch,dev]`.
This creates an virtual environment of about 703M in size.

```bash
git clone https://github.com/nuru-nu/nucas &&
cd nucas &&
python -m venv env &&
. env/bin/activate &&
pip install --upgrade pip &&
pip install -e .[pytorch,dev]
```

TODO: add jupyter support

TODO: add scripts

TODO: accelerate training with metal

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
db.save(config, model, stats, imgs, overwrite=True)
```

### Run in Browser

TODO: convert ncas for ca.js

TODO: reimplement ca.js with swiss.gl

TODO: build server for midi input / syphon output
