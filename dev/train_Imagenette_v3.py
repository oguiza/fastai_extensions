from fastai.script import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.distributed import *
from fastprogress import fastprogress
from torchvision.models import *
#from fastai.vision.models.xresnet import *
#from fastai.vision.models.xresnet2 import *
#from fastai.vision.models.presnet import *
from exp.nb_xresnet_sa import *
torch.backends.cudnn.benchmark = True
fastprogress.MAX_COLS = 80


def get_data(size=128, woof=0, use_partial=1., bs=256, workers=None, tfm_sch=0, blsize=0., blalpha=0., bltype='mix', blgrid=True, blsame=True, blproba=False):
    if   size<=128: path = URLs.IMAGEWOOF_160 if woof else URLs.IMAGENETTE_160
    elif size<=224: path = URLs.IMAGEWOOF_320 if woof else URLs.IMAGENETTE_320
    else          : path = URLs.IMAGEWOOF     if woof else URLs.IMAGENETTE
    path = untar_data(path)
    n_gpus = num_distrib() or 1
    if workers is None: workers = min(8, num_cpus()//n_gpus)
    if blsize != 0 and tfm_sch == 0:
        from exp.nb_new_data_augmentation import blend
        tfms = ([flip_lr(p=0.5), blend(size=blsize, alpha=blalpha, blend=bltype, grid=blgrid, same=blsame, proba=blproba, p=1.)], [])
    else: tfms = ([flip_lr(p=0.5)], [])
    return (ImageList.from_folder(path)
            .use_partial_data(use_partial)
            .split_by_folder(valid='val')
            .label_from_folder().transform(tfms, size=size)
            .databunch(bs=bs, num_workers=workers)
            .presize(size, scale=(0.35,1))
            .normalize(imagenet_stats))

@call_parse
def main(
    gpu:Param("GPU to run on", str)=None,
    use_partial: Param("Use partial data", float)=1.,
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
    cutmix: Param("Cutmix", float)=0.,
    blend: Param("Blend", float)=0.,
    opt: Param("Optimizer (adam,rms,sgd)", str)='adam',
    arch: Param("Architecture (xresnet34, xresnet50, presnet34, presnet50)", str)='xresnet50',
    blsize: Param("Blend size", float)=0.,
    blalpha: Param("Blend alpha", float)=0.1,
    bltype: Param("Blend type", str)=None,
    blgrid: Param("Blend grid", int)=1,
    blsame: Param("Blend same", int)=1,
    blproba: Param("Blend proba", int)=0,
    sa: Param("Self-attention", int)=0,
    sym: Param("Symmetry for self-attention", int)=0,
    tfm_sch: Param("Transform Scheduler", int)=0,
    sch_test: Param("Scheduler test", int)=0,
    sch_param: Param("Scheduler param", str)='alpha',
    sch_val_min: Param("Scheduler param min", float)=0.,
    sch_val_max: Param("Scheduler param max", float)=.5,
    sch_iter_min: Param("Scheduler param min", float)=0.,
    sch_iter_max: Param("Scheduler param max", float)=1.,
    sch_func: Param("Scheduler func", str)='cos',
    plot: Param("Plot Scheduler", int)=1,
    dump: Param("Print model; don't train", int)=0,
            ):
    "Distributed training of Imagenette."
    n_gpus = torch.cuda.device_count()
    print('n_gpus:', n_gpus)

    gpu = setup_distrib(gpu)
    if gpu is None: bs *= max(1, torch.cuda.device_count())
    if   opt=='adam' : opt_func = partial(optim.Adam, betas=(mom,alpha), eps=eps)
    elif opt=='rms'  : opt_func = partial(optim.RMSprop, alpha=alpha, eps=eps)
    elif opt=='sgd'  : opt_func = partial(optim.SGD, momentum=mom)
    
    data = get_data(size=size, woof=woof, use_partial=use_partial, bs=bs, tfm_sch=tfm_sch, blsize=blsize, blalpha=blalpha, bltype=bltype, blgrid=blgrid, blsame=blsame, blproba=blproba)
    print(data.train_ds.tfms)
    bs_rat = bs/256
    if gpu is not None: bs_rat *= num_distrib()
    if not gpu: 
        print(f'lr: {lr}; eff_lr: {lr*bs_rat}; size: {size}; alpha: {alpha}; mom: {mom}; eps: {eps}; wd: {wd}')
        print(f'sa: {sa}; sym: {sym}')
    lr *= bs_rat
    m = globals()[arch]
    learn = (Learner(data, m(c_out=10, sa=sa, sym=sym), wd=wd, opt_func=opt_func,
    metrics=[accuracy,top_k_accuracy],
    bn_wd=False, true_wd=True,
    loss_func = LabelSmoothingCrossEntropy())
                )
    
    
    if dump: print(learn.model); exit()
    if not tfm_sch:
        if mixup: 
            print('mixup:', mixup)
            learn = learn.mixup(alpha=mixup)
        elif cutmix: 
            print('cutmix:', cutmix)
            alpha = cutmix
            from exp.nb_new_data_augmentation import cutmix
            learn = learn.cutmix(alpha=alpha)
        elif blend: 
            print('blend:', blend)
            from exp.nb_new_data_augmentation import blend
            learn = learn.blend(size=blsize, alpha=blalpha, blend=bltype, grid=blgrid, same=blsame, proba=blproba)
    elif tfm_sch: 
        from exp.nb_new_data_augmentation import TfmScheduler, cosine_annealing
        from fastai.callback import annealing_cos, annealing_linear
        if mixup: 
            print('sch_mixup:', mixup)
            alpha = mixup
            from fastai.callbacks import mixup
            tfm_fn = partial(mixup, alpha=alpha)
        elif blend:
            print('sch_blend:', blend)
            from exp.nb_new_data_augmentation import blend
            tfm_fn = partial(blend, size=blsize, alpha=blalpha, blend=bltype, grid=blgrid, same=blsame, proba=blproba)
        elif cutmix: 
            print('sch_cutmix:', cutmix)
            alpha = cutmix
            from exp.nb_new_data_augmentation import cutmix
            tfm_fn = partial(cutmix, alpha=alpha)
        sch_val = (sch_val_min, sch_val_max)
        sch_iter = (sch_iter_min, sch_iter_max)
        if sch_func is None or sch_func == 'cos': func = annealing_cos
        elif sch_func == 'bicos': func = cosine_annealing
        elif sch_func == 'lin': func = annealing_linear
        else: 
            print('Incorrect sch_func:', sch_func)
            return
        sch_tfm_cb = partial(TfmScheduler, tfm_fn=tfm_fn, sch_param=sch_param, 
                             sch_val=sch_val, sch_iter=sch_iter, sch_func=func, 
                             plot=plot, test=sch_test)
        learn.callback_fns.append(sch_tfm_cb)
        print(sch_tfm_cb)
    if n_gpus >0: 
        print('to_fp16: True')
        learn = learn.to_fp16(dynamic=True)
    if gpu is None:       learn.to_parallel()
    elif num_distrib()>1: learn.to_distributed(gpu) # Requires `-m fastai.launch`
    learn.fit_one_cycle(epochs, lr, div_factor=10, pct_start=0.3)
    try:
        learn.recorder.plot_losses()
        learn.recorder.plot_metrics()
    except: pass