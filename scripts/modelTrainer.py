import pytorch_lightning as pl
import torch
import os

def setSingleTrainer(max_epoch = 30, num_devices = None):
    """initialize the Trainer

    Args:
        max_epoch (int, optional): amount of maximum epoch to reach. Defaults to 30.
        num_devices (int, optional): amount of device to use for the trainer (only GPU). Defaults to None.

    Returns:
        Trainer: a pytorch lightning trainer
    """ 
    
    if torch.cuda.is_available():
        num_devices = num_devices if num_devices is not None else torch.cuda.device_count()
        accelerator = 'gpu'
    else:
        num_devices = 1
        accelerator = 'cpu'

    return pl.Trainer(
        max_epochs=max_epoch,
        accelerator=accelerator,
        log_every_n_steps=1,
        devices=num_devices
    )

def setAllTrainers(max_epoch_sem = 30, max_epoch_coarse = 30, max_epoch_fine = 30, num_devices = None):
    
    semantic_trainer = setSingleTrainer(max_epoch_sem, num_devices)
    coarse_trainer = setSingleTrainer(max_epoch_coarse, num_devices)
    fine_trainer = setSingleTrainer(max_epoch_fine, num_devices)
    
    return semantic_trainer, coarse_trainer, fine_trainer

def modelFit(model, trainer, train_loader, valid_loader = None, checkpoint_path = None, myDevice = "cpu"):
    
    """check if exists a train checkpoint and starts training

    Args:
        model (Decoder): instance of a Decoder Model
        trainer (Trainer): instance of Trainer
        train_loader (DataLoader): instance of the train dataloader
        valid_loader (DataLoader, optional): instance of the validation dataloader. Defaults to None.
        checkpoint_path (Path, optional): path to .ckpt file. Defaults to None.
    """
        
    model.train()
    if os.path.exists(checkpoint_path):
        print(f"Checkpoint found at {checkpoint_path}. Resuming training...")
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader, ckpt_path=checkpoint_path)
    else:
        print("No checkpoint found. Starting from scratch...")
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    
    model.to(myDevice)
