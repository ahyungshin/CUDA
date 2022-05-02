from .deeplab_multi import DeeplabMulti


def get_model(cfg):
    if cfg.model.backbone == "deeplabv2_multi":
        tm = False
        num_target = 2
        eval_target = 2
        model = DeeplabMulti(tm, num_target, eval_target,num_classes=cfg.data.num_classes, init=cfg.model.imagenet_pretrained)
        params = model.optim_parameters(lr=cfg.opt.lr)
    else:
        raise NotImplementedError()
    return model, params
