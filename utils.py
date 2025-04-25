"""Contains various functions for Pytorch model training and saving.
"""

import  torch
from pathlib import Path

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a Pytorch model to a target directory.
    Args:
        model: A Pytorch model to save.
        target_dir: A directory for saving the model to.
        model_name: A filename for the saved model. Should include
        either ".pth" or ".pt" as the file extension.
    Example usage:
    save_model(model=model,
               target_dir="models",
               model_name="05_going_modular_cell_10_percent.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswit(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model
    print(f"[INFO] Saving model to {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)
    

    # Create model save path
    
