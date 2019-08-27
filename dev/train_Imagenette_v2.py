from fastai.script import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.distributed import *
from fastprogress import fastprogress
from torchvision.models import *
from exp.nb_xresnet import *
from fastai.basic_train import Learner


torch.backends.cudnn.benchmark = True
fastprogress.MAX_COLS = 80

def get_data(size, woof, bs, workers=None, gridcut_tfm=None, gridcut_alpha=.4, randomgrid_tfm=None, randomgrid_alpha=.4, gridmix_tfm=None, gridmix_alpha=.4, grid_random=False, shuffle_ch=False):
    if size<=128: path = URLs.IMAGEWOOF_160 if woof else URLs.IMAGENETTE_160
    elif size<=224: path = URLs.IMAGEWOOF_320 if woof else URLs.IMAGENETTE_320
    else          : path = URLs.IMAGEWOOF     if woof else URLs.IMAGENETTE
    path = untar_data(path)
    n_gpus = num_distrib() or 1
    if workers is None: workers = min(8, num_cpus()//n_gpus)
    tfms = ([flip_lr(p=0.5)], [])
    if gridcut_tfm is not None: 
        from exp.nb_new_data_augmentation import gridcut
        tfms = ([flip_lr(p=0.5), gridcut(n_patches=gridcut_tfm, alpha=gridcut_alpha, random=grid_random, p=1.)], [])
    elif gridmix_tfm is not None: 
        from exp.nb_new_data_augmentation import gridmix
        if gridmix_alpha is None: gridmix_alpha = .2
        tfms = ([flip_lr(p=0.5), gridmix(n_patches=gridmix_tfm, alpha=gridmix_alpha, random=grid_random, p=1.)], [])
    elif randomgrid_tfm is not None: 
        from exp.nb_new_data_augmentation import randomgrid
        if randomgrid_alpha is None: randomgrid_alpha = .2
        tfms = ([flip_lr(p=0.5), randomgrid(n_patches=randomgrid_tfm, alpha=randomgrid_alpha, random=grid_random, p=1.)], [])
    if shuffle_ch:
        from exp.nb_new_data_augmentation import shuffle_ch
        tfms[0].append(shuffle_ch())
    print(tfms)
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
    epochs: Param("Number of epochs", int)=5,
    bs: Param("Batch size", int)=256,
    wd: Param("Weight decay", float)=.01,
    gridcut_tfm: Param("Gridcut", int)=None,
    gridcut_alpha: Param("Gridcut_alpha", float)=None,
    gridmix_tfm: Param("Gridmix", int)=None,
    gridmix_alpha: Param("Gridmix_alpha", float)=None,
    randomgrid_tfm: Param("RandomGrid", int)=None,
    randomgrid_alpha: Param("RandomGrid_alpha", float)=None,
    shuffle_ch: Param("Shuffle channels", int)=0,
    mixup: Param("Mixup", float)=0.,
    ricap: Param("Ricap", float)=0.,
    siricap: Param("Siricap", float)=0.,
    icap: Param("Icap", float)=0.,
    cutmix: Param("Cutmix", float)=0.,
    
    randcutmix: Param("Randcutmix", float)=0.,
    sirandcutmix: Param("Sirandcutmix", float)=0.,
    true_lambda: Param("Cutmix true_λ", int)=1,
    grid_random: Param("Grid random", int)=0,
    opt: Param("Optimizer (adam,rms,sgd)", str)='adam',
    arch: Param("Architecture (xresnet34, xresnet50, presnet34, presnet50)", str)='xresnet50',
    dump: Param("Print model; don't train", int)=0,
            ):
    "Distributed training of Imagenette."
    gpu = setup_distrib(gpu)
    if gpu is None: bs *= torch.cuda.device_count()
    if   opt=='adam' : opt_func = partial(optim.Adam, betas=(mom,alpha), eps=eps)
    elif opt=='rms'  : opt_func = partial(optim.RMSprop, alpha=alpha, eps=eps)
    elif opt=='sgd'  : opt_func = partial(optim.SGD, momentum=mom)
    if gridcut_tfm is not None:
        data = get_data(size, woof, bs, gridcut_tfm=gridcut_tfm, gridcut_alpha=gridcut_alpha, 
                       grid_random=grid_random, shuffle_ch=shuffle_ch)
    elif gridmix_tfm is not None:
        data = get_data(size, woof, bs, gridmix_tfm=gridmix_tfm, gridmix_alpha=gridmix_alpha,
                        grid_random=grid_random, shuffle_ch=shuffle_ch)
    elif randomgrid_tfm is not None:
        data = get_data(size, woof, bs, randomgrid_tfm=randomgrid_tfm, randomgrid_alpha=randomgrid_alpha,
                        grid_random=grid_random, shuffle_ch=shuffle_ch)
    else:
        data = get_data(size, woof, bs, shuffle_ch=shuffle_ch)
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
    if mixup: 
        print('mixup alpha=', mixup)
        learn = learn.mixup(alpha=mixup)
    elif ricap: 
        beta = ricap
        from exp.nb_new_data_augmentation import ricap
        learn = ricap(learn, beta=beta)
        print('ricap beta=', beta)
    elif icap: 
        beta = icap
        from exp.nb_new_data_augmentation import icap
        learn = icap(learn, beta=beta)
        print('icap beta=', beta)
    elif siricap: 
        beta = siricap
        from exp.nb_new_data_augmentation import siricap
        learn = siricap(learn, beta=beta)
        print('siricap beta=', beta)
    elif cutmix: 
        alpha = cutmix
        from exp.nb_new_data_augmentation import cutmix
        learn = cutmix(learn, alpha=alpha, true_λ=true_lambda)
        print('cutmix alpha=', alpha, 'true_lambda=', true_lambda)
    elif sirandcutmix: 
        alpha = sirandcutmix
        from exp.nb_new_data_augmentation import sirandcutmix
        learn = sirandcutmix(learn, alpha=alpha, true_λ=true_lambda)
        print('sirandcutmix alpha=', alpha, 'true_lambda=', true_lambda)
    elif randcutmix: 
        alpha = randcutmix
        from exp.nb_new_data_augmentation import randcutmix
        learn = randcutmix(learn, alpha=alpha, true_λ=true_lambda)
        print('randcutmix alpha=', alpha, 'true_lambda=', true_lambda)
    learn = learn.to_fp16(dynamic=True)
    if gpu is None:       learn.to_parallel()
    elif num_distrib()>1: learn.to_distributed(gpu) # Requires `-m fastai.launch`
    learn.fit_one_cycle(epochs, lr, div_factor=10, pct_start=0.3)
    learn.recorder.plot_losses()
    learn.recorder.plot_metrics()