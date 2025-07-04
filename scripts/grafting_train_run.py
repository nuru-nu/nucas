import numpy as np

import nucas

db = nucas.db.get_db()

config = nucas.train.get_config()
config.target = 'https://www.robots.ox.ac.uk/~vgg/data/dtd/images/chequered/chequered_0045.jpg'
config.steps = 1000
model1, stats = nucas.train.train(config)
imgs = nucas.run.run(config, model1)
db.save(config, model1, stats, imgs, overwrite=True)

config2 = nucas.train.get_config()
config2.target = (
    'https://www.robots.ox.ac.uk/~vgg/data/dtd/images/dotted/dotted_0112.jpg'
)
config2.steps = 2000
config2.parent = config.id
model2, stats2 = nucas.train.train(config2)
imgs = nucas.run.run(config2, model2)
db.save(config2, model2, stats, imgs, overwrite=True)


# Graft the first model onto the second
s = 10
sz = 256

update_rate_f = nucas.utils.make_f(lambda x: x, [0.5, 0.5])
per_step_f = nucas.utils.make_f(lambda x: 1 + x * 15, [0, 1])

mask_f = lambda x: 1 / (
    1
    + np.exp(-np.linspace(-s, s, sz) + (x - 0.5) * 15)[None]
    + np.zeros([sz, 1])
)

imgs = nucas.run.run_graft(
    config,
    model1,
    model2,
    steps=3000,
    cycles=1,
    mask_f=mask_f,
    update_rate_f=update_rate_f,
    per_step_f=per_step_f,
    sz=sz,
)

print(f'Grafted {len(imgs)} images')
db.save_video(config, imgs, overwrite=True)
