"""Smoke screen script test that trains a single NCA and stores it in the db."""

import nucas

config = nucas.train.get_config()
config.target = 'https://www.robots.ox.ac.uk/~vgg/data/dtd/images/chequered/chequered_0045.jpg'

model, stats = nucas.train.train(config)
imgs = nucas.run.run(config, model)

db = nucas.db.get_db()
db.save(config, model, stats, imgs, overwrite=True)
