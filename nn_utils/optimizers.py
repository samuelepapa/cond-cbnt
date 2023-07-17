import torch


def get_optimizer(optimizer_config, parameters, lr):
    # Optimizer
    if optimizer_config.type == "adam":
        optimizer = torch.optim.Adam(
            parameters,
            weight_decay=optimizer_config.weight_decay,
            lr=lr,
            betas=optimizer_config.betas,
            eps=optimizer_config.eps,
        )
    else:
        raise NotImplementedError(f"Optimizer {optimizer_config.type} not implemented.")
    return optimizer
