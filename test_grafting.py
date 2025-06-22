import logging, sys
import numpy as np
import nucas

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)

db = nucas.db.get_db()

config = nucas.train.get_config()
config.target = 'https://www.robots.ox.ac.uk/~vgg/data/dtd/images/chequered/chequered_0045.jpg'
config.steps = 1000
model1, stats = nucas.train.train(config)
imgs = nucas.run.run(config, model1)
db.save(config, model1, stats, imgs, overwrite=True)

config = nucas.train.get_config()
config.target = 'https://www.robots.ox.ac.uk/~vgg/data/dtd/images/dotted/dotted_0112.jpg'
config.steps = 2000
model2, stats2 = nucas.train.train(config)
imgs = nucas.run.run(config, model2)
db.save(config, model2, stats, imgs, overwrite=True)


# Graft the first model onto the second
s = 10
sz = 256

update_rate_f = nucas.utils.make_f(
    lambda x: x,
    [.5, .5])
per_step_f = nucas.utils.make_f(
    lambda x: 1 + x*15,
    [0, 1])

mask_f = lambda x: 1 / (1 + np.exp(-np.linspace(-s, s , sz) + (x-.5)*15)[None] + np.zeros([sz, 1]))

imgs = nucas.run.run_graft(
    config, model1, model2, steps=3000, cycles=1,
    mask_f=mask_f, update_rate_f=update_rate_f, per_step_f=per_step_f, sz=sz)

print(f'Grafted {len(imgs)} images')
db.save_video(config, imgs, overwrite=True)
