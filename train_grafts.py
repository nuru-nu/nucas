import logging, sys
import numpy as np
import nucas

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)

db = nucas.db.get_db()

parent_target = {
  "img": 'https://www.robots.ox.ac.uk/~vgg/data/dtd/images/chequered/chequered_0045.jpg',
  "steps": 1000,
  "child_targets": [
    {
      "img": 'https://www.robots.ox.ac.uk/~vgg/data/dtd/images/dotted/dotted_0112.jpg',
      "steps": 2000,
    },
    {
      # "img": 'https://www.robots.ox.ac.uk/~vgg/data/dtd/images/dotted/dotted_0112.jpg', # Example of a second child target
      # "steps": 2000,
    }
  ]
}

def train_model(target, parent=None):
    config = nucas.train.get_config()
    config.target = target['img']
    config.steps = target['steps']
    config.parent = parent.id if parent else None
    model, stats = nucas.train.train(config)
    imgs = nucas.run.run(config, model)
    db.save(config, model, stats, imgs, overwrite=True)
    logging.info(f"Model trained: {config.id}")
    logging.info(f"Saved model with ID: {config.id}")
    return config, model

# Train the parent model
parent_config, parent_model = train_model(parent_target)

# Train the child models
trained_children = []
for child_target in parent_target['child_targets']:
    config, model = train_model(child_target, parent_config)
    trained_children.append(
        {
            'config': config,
            'model': model
        }
    )

# child ids to txt file with parent id as name
with open(db.path(f'{parent_config.id}_children.txt'), 'w') as f:
    for child in trained_children:
        logging.info(f"Child model trained: {child['config'].id}")
        f.write(f"{child['config'].id}\n")
