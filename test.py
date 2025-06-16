import logging, sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)


import nucas

config = nucas.train.get_config()
config.target = 'https://www.robots.ox.ac.uk/~vgg/data/dtd/images/chequered/chequered_0045.jpg'
config.steps = 2

model, stats = nucas.train.train(config)
imgs = nucas.run.run(config, model)

db = nucas.db.get_db()
db.save(config, model, stats, imgs, overwrite=True)
