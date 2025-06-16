import sys

if "google.colab" in sys.modules:
  from . import colab

else:
  from . import colab_shim as colab

max_output_height = colab.max_output_height
RegionSelector = colab.RegionSelector
ImageAndGraph = colab.ImageAndGraph
init = colab.init
