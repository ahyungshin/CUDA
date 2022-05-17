from .deeplab_multi import DeeplabMulti


def get_model(cfg):
    if cfg.model.backbone == "deeplabv2_multi":
        model = DeeplabMulti(cfg.tm, cfg.num_target, cfg.eval_target, num_classes=cfg.data.num_classes, init=cfg.model.imagenet_pretrained)
        params = model.optim_parameters(cfg)
    else:
        raise NotImplementedError()
    return model, params
