import torch


def get_scheduler(optimizer, scheduler_config):
    schedulers = []
    if scheduler_config.warmup.do:
        schedulers.append(
            torch.optim.lr_scheduler.LinearLR(
                optimizer=optimizer,
                start_factor=optimizer.param_groups[0]["lr"]
                * scheduler_config.warmup.start_factor,
                total_iters=scheduler_config.warmup.num_steps,
            )
        )
    if scheduler_config.step.do:
        schedulers.append(
            torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_config.step.step_size,
                gamma=scheduler_config.step.gamma,
            )
        )
    return torch.optim.lr_scheduler.ChainedScheduler(schedulers)
