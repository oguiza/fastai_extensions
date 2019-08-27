from fastai.script import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.distributed import *
from fastprogress import fastprogress
from torchvision.models import *
#from fastai.vision.models.xresnet import *
#from fastai.vision.models.xresnet2 import *
#from fastai.vision.models.presnet import *
torch.backends.cudnn.benchmark = True
fastprogress.MAX_COLS = 80

import torch.nn as nn
import torch,math,sys
import torch.utils.model_zoo as model_zoo
from functools import partial
from fastai.torch_core import Module

__all__ = ['XResNet', 'xresnet18', 'xresnet34', 'xresnet50', 'xresnet101', 'xresnet152']

# or: ELU+init (a=0.54; gain=1.55)
act_fn = nn.ReLU(inplace=True)

class Flatten(Module):
    def forward(self, x): return x.view(x.size(0), -1)

def init_cnn(m):
    if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv2d,nn.Linear)): nn.init.kaiming_normal_(m.weight)
    for l in m.children(): init_cnn(l)

def conv(ni, nf, ks=3, stride=1, bias=False):
    return nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=ks//2, bias=bias)

def noop(x): return x

def conv_layer(ni, nf, ks=3, stride=1, zero_bn=False, act=True):
    bn = nn.BatchNorm2d(nf)
    nn.init.constant_(bn.weight, 0. if zero_bn else 1.)
    layers = [conv(ni, nf, ks, stride=stride), bn]
    if act: layers.append(act_fn)
    return nn.Sequential(*layers)

class ResBlock(Module):
    def __init__(self, expansion, ni, nh, stride=1):
        nf,ni = nh*expansion,ni*expansion
        layers  = [conv_layer(ni, nh, 3, stride=stride),
                   conv_layer(nh, nf, 3, zero_bn=True, act=False)
        ] if expansion == 1 else [
                   conv_layer(ni, nh, 1),
                   conv_layer(nh, nh, 3, stride=stride),
                   conv_layer(nh, nf, 1, zero_bn=True, act=False)
        ]
        self.convs = nn.Sequential(*layers)
        # TODO: check whether act=True works better
        self.idconv = noop if ni==nf else conv_layer(ni, nf, 1, act=False)
        self.pool = noop if stride==1 else nn.AvgPool2d(2, ceil_mode=True)

    def forward(self, x): return act_fn(self.convs(x) + self.idconv(self.pool(x)))

def filt_sz(recep): return min(64, 2**math.floor(math.log2(recep*0.75)))

class XResNet(nn.Sequential):
    def __init__(self, expansion, layers, c_in=3, c_out=1000):
        stem = []
        sizes = [c_in,32,32,64]
        for i in range(3):
            stem.append(conv_layer(sizes[i], sizes[i+1], stride=2 if i==0 else 1))
            #nf = filt_sz(c_in*9)
            #stem.append(conv_layer(c_in, nf, stride=2 if i==1 else 1))
            #c_in = nf

        block_szs = [64//expansion,64,128,256,512]
        blocks = [self._make_layer(expansion, block_szs[i], block_szs[i+1], l, 1 if i==0 else 2)
                  for i,l in enumerate(layers)]
        super().__init__(
            *stem,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            *blocks,
            nn.AdaptiveAvgPool2d(1), Flatten(),
            nn.Linear(block_szs[-1]*expansion, c_out),
        )
        init_cnn(self)

    def _make_layer(self, expansion, ni, nf, blocks, stride):
        return nn.Sequential(
            *[ResBlock(expansion, ni if i==0 else nf, nf, stride if i==0 else 1)
              for i in range(blocks)])

def xresnet(expansion, n_layers, name, pretrained=False, **kwargs):
    model = XResNet(expansion, n_layers, **kwargs)
    if pretrained: model.load_state_dict(model_zoo.load_url(model_urls[name]))
    return model

me = sys.modules[__name__]
for n,e,l in [
    [ 18 , 1, [2,2,2 ,2] ],
    [ 34 , 1, [3,4,6 ,3] ],
    [ 50 , 4, [3,4,6 ,3] ],
    [ 101, 4, [3,4,23,3] ],
    [ 152, 4, [3,8,36,3] ],
]:
    name = f'xresnet{n}'
    setattr(me, name, partial(xresnet, expansion=e, n_layers=l, name=name))
    

def get_data(size=128, woof=0, bs=256, workers=None, blsize=0., blproba=0., bltype='mix', blgrid=True, blsame=True):
    if   size<=128: path = URLs.IMAGEWOOF_160 if woof else URLs.IMAGENETTE_160
    elif size<=224: path = URLs.IMAGEWOOF_320 if woof else URLs.IMAGENETTE_320
    else          : path = URLs.IMAGEWOOF     if woof else URLs.IMAGENETTE
    path = untar_data(path)
    n_gpus = num_distrib() or 1
    if workers is None: workers = min(8, num_cpus()//n_gpus)
    if blsize != 0:
        from exp.nb_new_data_augmentation import blender
        tfms = ([flip_lr(p=0.5), blender(size=blsize, proba=blproba, blend=bltype, grid=blgrid, same=blsame, p=1.)], [])
    else: tfms = ([flip_lr(p=0.5)], [])
    return (ImageList.from_folder(path).split_by_folder(valid='val')
                .label_from_folder().transform(tfms, size=size)
                .databunch(bs=bs, num_workers=workers)
                .presize(size, scale=(0.35,1))
                .normalize(imagenet_stats))

@call_parse
def main(
    gpu:Param("GPU to run on", str)=None,
    woof: Param("Use imagewoof (otherwise imagenette)", int)=0,
    lr: Param("Learning rate", float)=1e-3,
    size: Param("Size (px: 128,192,224)", int)=128,
    alpha: Param("Alpha", float)=0.99,
    mom: Param("Momentum", float)=0.9,
    eps: Param("epsilon", float)=1e-6,
    wd: Param("Weight decay", float)=0.01,
    epochs: Param("Number of epochs", int)=5,
    bs: Param("Batch size", int)=256,
    mixup: Param("Mixup", float)=0.,
    opt: Param("Optimizer (adam,rms,sgd)", str)='adam',
    arch: Param("Architecture (xresnet34, xresnet50, presnet34, presnet50)", str)='xresnet50',
    blsize: Param("Blender size", float)=0.1,
    blproba: Param("Blender size", float)=0.1,
    bltype: Param("Blender type", str)=None,
    blgrid: Param("Blender grid", int)=1,
    blsame: Param("Blender same", int)=1,
    dump: Param("Print model; don't train", int)=0,
            ):
    "Distributed training of Imagenette."

    gpu = setup_distrib(gpu)
    if gpu is None: bs *= max(1, torch.cuda.device_count())
    if   opt=='adam' : opt_func = partial(optim.Adam, betas=(mom,alpha), eps=eps)
    elif opt=='rms'  : opt_func = partial(optim.RMSprop, alpha=alpha, eps=eps)
    elif opt=='sgd'  : opt_func = partial(optim.SGD, momentum=mom)
    
    data = get_data(size=size, woof=woof, bs=bs, blsize=blsize, blproba=blproba, bltype=bltype, blgrid=blgrid, blsame=blsame)
    print(data.train_ds.tfms)
    bs_rat = bs/256
    if gpu is not None: bs_rat *= num_distrib()
    if not gpu: print(f'lr: {lr}; eff_lr: {lr*bs_rat}; size: {size}; alpha: {alpha}; mom: {mom}; eps: {eps}; wd: {wd}')
    lr *= bs_rat
    m = globals()[arch]
    learn = (Learner(data, m(c_out=10), wd=wd, opt_func=opt_func,
    metrics=[accuracy,top_k_accuracy],
    bn_wd=False, true_wd=True,
    loss_func = LabelSmoothingCrossEntropy())
                )
    if dump: print(learn.model); exit()
    if mixup: learn = learn.mixup(alpha=mixup)
    learn = learn.to_fp16(dynamic=True)
    if gpu is None:       learn.to_parallel()
    elif num_distrib()>1: learn.to_distributed(gpu) # Requires `-m fastai.launch`
    learn.fit_one_cycle(epochs, lr, div_factor=10, pct_start=0.3)
    learn.recorder.plot_losses()
    learn.recorder.plot_metrics()