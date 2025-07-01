import logging, sys
import numpy as np
import nucas
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)

db = nucas.db.get_db()

def run_graft(config, model1, model2, nr_frames=3000, cycles=1, s=10, sz=256):
    update_rate_f = nucas.utils.make_f(
        lambda x: x,
        [.5, .5])
    per_step_f = nucas.utils.make_f(
        lambda x: 1 + x*15,
        [0, 1])

    mask_f = lambda x: 1 / (1 + np.exp(-np.linspace(-s, s , sz) + (x-.5)*15)[None] + np.zeros([sz, 1]))

    imgs = nucas.run.run_graft(
        config, model1, model2, steps=nr_frames, cycles=cycles,
        mask_f=mask_f, update_rate_f=update_rate_f, per_step_f=per_step_f, sz=sz)

    print(f'Grafted {len(imgs)} images')
    db.save_video(config, imgs, overwrite=True)
    return config


child_list_path = '/Users/groux/ncas/20250701_221523_children.txt'

# Read parent ID and child IDs from file
parent_id = os.path.basename(child_list_path).replace('_children.txt', '')
child_ids = []

with open(child_list_path) as f:
  child_ids = [line.strip() for line in f.readlines()]

# Load models and run grafts
parent_config, parent_model, _ = db.load(parent_id)

for child_id in child_ids:
  child_config, child_model, _ = db.load(child_id)
  run_graft(child_config, parent_model, child_model)
