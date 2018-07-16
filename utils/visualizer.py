import visdom
import time
import numpy as np
from torchvision.utils import make_grid


class Visualizer(object):

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = dict()
        self.log_text = str()

    def re_init(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)

    def plot_many(self, dict):
        for key, val in dict.items():
            self.plot(key, val)

    def plot(self, name, y, **kwargs):
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      win=name,
                      **kwargs)
        self.index[name] = x + 1

    def img_many(self, dict):
        for key, val in dict:
            self.img(key, val)

    def img(self, name, img, **kwargs):

        self.vis.images(img.to('cpu').numpy(),
                        win=name,
                        opts=dict(title=name),
                        **kwargs)

    def img_grid_many(self, dict):
        for key, val in dict.items():
            self.img_grid(key, val)

    def img_grid(self, name, input_3d):
        """
        一个batch的图片转成一个网格图，i.e. input（36，64，64）
        会变成 6*6 的网格图，每个格子大小64*64
        """
        self.img(name, make_grid(input_3d.cpu()[0].unsqueeze(1).clamp(max=1, min=0)))

    def log(self, info, win='log_text'):

        self.log_text += ('[{time}] {info} <br>'.format(time=time.strftime('%m%d_%H%M%S'), info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        return getattr(self.vis, name)
