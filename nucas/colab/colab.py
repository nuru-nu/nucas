import base64
import io
import json
import logging
import os
import sys

import IPython
import IPython.display
import google.colab
import numpy as np
import PIL

from .. import utils


def max_output_height(pixels):

  IPython.display.display(
      IPython.display.Javascript(
          'google.colab.output.setIframeHeight(0, true, {maxHeight: %d})'
          % pixels
      )
  )


class RegionSelector:

  def __init__(self, url, maxhw=800):
    img = utils.imread(url)
    scale = min(1.0, maxhw / max(*img.size))
    src = self._img2src(img.resize([int(_ * scale) for _ in img.size]))
    IPython.display.display(
        IPython.display.HTML(
            (
                r"""
    <pre id="stuff">
      <img id="img" src="__SRC__">
    </pre>
    <pre id="pre"></pre>
    <script>
    google.colab.output.setIframeHeight(0, true, {maxHeight: 9999})
    const img = document.getElementById('img')
    const pre = document.getElementById('pre')
    const stuff = document.getElementById('stuff')
    const boxes = []
    const bcr = img.getBoundingClientRect()
    const scale = __SCALE__
    const url = __URL__
    console.log('bcr', bcr);
    const el = document.createElement('div')
    let x0=null, y0=null
    function update(x, y) {
      const x1 = Math.min(x, x0)
      const y1 = Math.min(y, y0)
      const x2 = Math.max(x, x0)
      const y2 = Math.max(y, y0)
      const el = boxes[boxes.length - 1]
      el.style.left = x1 + 'px'
      el.style.top = y1 + 'px'
      el.style.width = (x2 - x1) + 'px'
      el.style.height = (y2 - y1) + 'px'
      const ret = [x1 - bcr.x, y1 - bcr.y, x2 - bcr.x, y2 - bcr.y]
      return ret.map(x => Math.round(x / scale))
    }
    img.addEventListener('mousedown', e => {
      el.style.border = '2px solid red'
      el.style.position = 'absolute'
      stuff.append(el)
      boxes.push(el)
      x0 = e.clientX
      y0 = e.clientY
      update(x0, y0)
      e.stopPropagation()
    })
    window.addEventListener('mousemove', e => {
      if (x0 === null) return
      update(e.clientX, e.clientY)
      e.stopPropagation()
    })
    window.addEventListener('mouseup', e => {
      if (x0 === null) return
      const coords = update(e.clientX, e.clientY)
      pre.textContent = `
config.target = ${JSON.stringify(url)}
config.crop_region = ${JSON.stringify(coords)}`
      x0 = y0 = null
      e.stopPropagation()
    })
    """
            )
            .replace('__SRC__', src)
            .replace('__SCALE__', str(scale))
            .replace('__URL__', json.dumps(url))
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
    google.colab.output.eval_js(
        f'TextImage_set_img({self._img2src(img)})',
        ignore_result=self.ignore_result,
    )


class ImageAndGraph:

  def __init__(self, width=400, height=300):
    IPython.display.display(
        IPython.display.HTML(
            """
<script src="https://cdn.plot.ly/plotly-2.24.1.min.js" charset="utf-8"></script>
<img id="img">
<div id="canvas" style="width:__width__px;height:__height__px;"></div>
<script>
const img = document.querySelector('#img');
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
Plotly.newPlot('canvas', data, layout, config);

function ImageAndGraph_update_graph(x, y) {
  data[0].x = x;
  data[0].y = y;
  Plotly.react('canvas', data, layout, config);
}
function ImageAndGraph_set_img(src) {
  img.src = src;
}
</script>
""".replace(
                '__width__', str(width)
            ).replace(
                '__height__', str(height)
            )
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
    google.colab.output.eval_js(
        f'ImageAndGraph_set_img("{self._img2src(img)}")', ignore_result=True
    )

  def set_graph(self, x, y):
    """Updates graph data."""
    google.colab.output.eval_js(
        f'ImageAndGraph_update_graph({json.dumps(list(x))}, {json.dumps(list(y))})',
        ignore_result=True,
    )


def init(drive=True):
  if drive:
    utils.set_basedir('/gdrive/MyDrive/ncas')
    google.colab.drive.mount('/gdrive')
    os.makedirs(utils.get_basedir(), exist_ok=True)
