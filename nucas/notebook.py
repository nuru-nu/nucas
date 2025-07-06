import base64
import datetime
import enum
import io
import json
import logging
import os
import sys

import IPython
import IPython.display
import numpy as np
import PIL

from . import utils


def expand_vertically():
  """Disables annoying vertical scroll."""

  IPython.display.display(
      IPython.display.HTML(
          """
          <div id="el__TS__"></div>
          <script>
          console.log('script')
          if (window.google) {
            google.colab.output.setIframeHeight(0, true, {maxHeight: 9999})
          } else {
            const el = document.getElementById('el__TS__')
            console.log(el);
            el.closest('.jp-Cell-outputArea').style.maxHeight = '9999px'
          }
          </script>
  """.replace(
              '__TS__', datetime.datetime.now().strftime('%H%M%S_%f')
          )
      )
  )


class Environment(enum.Enum):
  """What environment are we running in?"""

  UNKNOWN = 'unknown'
  JUPYTER = 'jupyer'
  COLAB = 'colab'


# Also provide a handy nucas.notebook.tqdm shortcut:
if 'google.colab' in sys.modules:
  import tqdm.notebook as tqdm

  environment = Environment.COLAB
elif 'IPython' in sys.modules:
  import tqdm.notebook as tqdm

  environment = Environment.JUPYTER
else:
  import tqdm

  environment = Environment.UNKNOWN


class RegionSelector:

  def __init__(self, url, maxhw=800):
    img = utils.imread(url)
    scale = min(1.0, maxhw / max(*img.size))
    src = self._img2src(img.resize([int(_ * scale) for _ in img.size]))
    IPython.display.display(
        IPython.display.HTML(
            (
                r"""
    <div id="RegionSelector__TS__">
      <div class="wrapper" style="position: relative;">
        <div class="box" style="border: 2px solid red; position: absolute; pointer-events: none;"></div>
        <img id="img" src="__SRC__">
      </div>
      <pre id="pre"></pre>
    </div>
    <script>
    (function () {
      const el = document.getElementById('RegionSelector__TS__')
      const wrapper = el.querySelector('.wrapper')
      const box = el.querySelector('.box')
      const img = el.querySelector('img')
      const pre = el.querySelector('pre')
      const scale = __SCALE__
      const url = __URL__
      wrapper.append(box);
      let x0 = null, y0 = null
      function update(x, y) {
        const x1 = Math.min(x, x0)
        const y1 = Math.min(y, y0)
        const x2 = Math.max(x, x0)
        const y2 = Math.max(y, y0)
        box.style.left = x1 + 'px'
        box.style.top = y1 + 'px'
        box.style.width = (x2 - x1) + 'px'
        box.style.height = (y2 - y1) + 'px'
        const scale2 = img.width / img.naturalWidth;
        const ret = [x1, y1, x2, y2]
        return ret.map(x => Math.round(x / scale / scale2))
      }
      img.addEventListener('mousedown', e => {
        x0 = e.offsetX
        y0 = e.offsetY
        update(x0, y0)
        e.stopPropagation()
        e.preventDefault()
      })
      img.addEventListener('mousemove', e => {
        if (x0 === null) return
        update(e.offsetX, e.offsetY)
        e.stopPropagation()
        e.preventDefault()
      })
      img.addEventListener('mouseup', e => {
        if (x0 === null) return
        const coords = update(e.offsetX, e.offsetY)
        pre.textContent = `
  config.target = ${JSON.stringify(url)}
  config.crop_region = ${JSON.stringify(coords)}`
        x0 = y0 = null
        e.stopPropagation()
        e.preventDefault()
      })
      if (window.google) {
        google.colab.output.setIframeHeight(0, true, {maxHeight: 9999})
      } else {
        el.closest('.jp-Cell-outputArea').style.maxHeight = '9999px'
      }
    })();
    """
            )
            .replace('__SRC__', src)
            .replace('__SCALE__', str(scale))
            .replace('__URL__', json.dumps(url))
            .replace('__TS__', datetime.datetime.now().strftime('%H%M%S_%f'))
        )
    )

  def _img2src(self, img):
    if isinstance(img, np.ndarray):
      if img.dtype != 'uint8':
        img = (255 * img).astype('uint8')
      img = PIL.Image.fromarray(img)
    f = io.BytesIO()
    img.save(f, format='JPEG')
    s = base64.b64encode(f.getvalue()).decode('ascii')
    return 'data:image/jpg;base64,%s' % s


class ImageAndGraph:

  def __init__(self, width=400, height=300):
    self.ts = datetime.datetime.now().strftime('%H%M%S_%f')
    IPython.display.display(
        IPython.display.HTML(
            """
<script src="https://cdn.plot.ly/plotly-2.24.1.min.js" charset="utf-8"></script>  <!-- colab -->
<div id="ImageAndGraph__TS__">
  <img>
  <div class="canvas" style="width:__WIDTH__px;height:__HEIGHT__px;"></div>  <!-- jupyter -->
</div>
<script>
const ImageAndGraph__TS__ = (function() {
  const el = document.getElementById('ImageAndGraph__TS__')
  const img = el.querySelector('img');
  const canvas = el.querySelector('.canvas');
  const data = [{
      x: [],
      y: [],
      type: 'scatter',
      mode: 'markers',
      marker: {
        opacity: 0.3 // Set the opacity (alpha) value of markers to 0.1
      },
  }];
  const layout = {
      margin: { t: 0 },
      xaxis: { title: 'step' },
      yaxis: { title: 'loss', type: 'log' },
  };
  const config = {
      displayModeBar: false,
  };

  const script = el.querySelector('.plotly')
  if (script) {
    script.onload = () => {
      console.log('loaded')
      Plotly.newPlot(canvas, data, layout, config);
    }
  }

  return {
    update_graph: (x, y) => {
      data[0].x = x;
      data[0].y = y;
      Plotly.react(canvas, data, layout, config);
    },
    set_img: (src) => {
      img.src = src;
    },
  }

})()
function colab_update_graph(x, y) { ImageAndGraph__TS__.update_graph(x, y) }
function colab_set_img(img) { ImageAndGraph__TS__.set_img(img) }
</script>
""".replace(
                '__WIDTH__', str(width)
            )
            .replace('__HEIGHT__', str(height))
            .replace('__TS__', self.ts)
        )
    )

  def _img2src(self, img):
    if isinstance(img, np.ndarray):
      if img.dtype != 'uint8':
        img = (255 * img).astype('uint8')
      img = PIL.Image.fromarray(img)
    f = io.BytesIO()
    img.save(f, format='JPEG')
    s = base64.b64encode(f.getvalue()).decode('ascii')
    return 'data:image/jpg;base64,%s' % s

  def set_image(self, img):
    """Updates image content."""
    if environment == Environment.COLAB:
      import google.colab

      google.colab.output.eval_js(
          f'colab_set_img("{self._img2src(img)}")',
          ignore_result=True,
      )
    if environment == Environment.JUPYTER:
      IPython.display.display(
          IPython.display.Javascript(
              f'ImageAndGraph{self.ts}.set_img("{self._img2src(img)}")',
          )
      )

  def set_graph(self, x, y):
    """Updates graph data."""
    if environment == Environment.COLAB:
      import google.colab

      google.colab.output.eval_js(
          f'colab_update_graph({json.dumps(list(x))}, {json.dumps(list(y))})',
          ignore_result=True,
      )
    if environment == Environment.JUPYTER:
      IPython.display.display(
          IPython.display.Javascript(
              f'ImageAndGraph{self.ts}.update_graph({json.dumps(list(x))}, {json.dumps(list(y))})',
          )
      )


def init(drive=True):
  """Initializes notebook environment."""
  if drive:
    if environment == Environment.COLAB:
      import google.colab

      utils.set_basedir('/gdrive/MyDrive/ncas')
      google.colab.drive.mount('/gdrive')
      os.makedirs(utils.get_basedir(), exist_ok=True)
    else:
      logging.warning('Google Drive can only be mounted in Colab.')
