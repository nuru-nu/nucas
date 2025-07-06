# Set this up first so module initialization can use logging.
import logging, sys

if "google.colab" in sys.modules:
  for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


from . import utils
from . import db
from . import notebook
from . import backend
from . import train
from . import run


__all__ = [
    "backend",
    "db",
    "notebook",
    "run",
    "train",
    "utils",
]
__version__ = "0.1.0"
