"""Trains a Pytorch image classification model using device-agnostic code.
"""

import os
import torch
import multiprocessing

import data_setup, engine, model_builder, utils

from torchvision import transforms

# Set multiprocessing start method to 'spawn' for Windows compatibility
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    # Setup hyperparameters
    NUM_EPOCHS = 5
    BATCH_SIZE = 32
    HIDDEN_UNITS = 10
    LEARNING_RATE = 0.001

    # Setup directories
    train_dir = "data/pizza_steak_sushi/train"
    test_dir = "data/pizza_steak_sushi/test"

    # Setup target device and display GPU information
    print("\n=== GPU Information ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("No GPU available, using CPU")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nTraining will be performed on: {device.upper()}\n")

    # Create transforms
    data_transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
    ])

    # Create DataLoader with help from data_setup.py
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transform,
        batch_size=BATCH_SIZE
    )

    # Create model with help from model_builder.py
    model = model_builder.TinyVGG(
        input_shape=3,
        hidden_units=HIDDEN_UNITS,
        output_shape=len(class_names)
    ).to(device)

    # Verify model is on correct device
    print(f"Model is on: {next(model.parameters()).device}")

    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=LEARNING_RATE)

    # Start training with help from engine.py
    engine.train(model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=NUM_EPOCHS,
                device=device)

    # Save the model with help from utils.py
    utils.save_model(model=model,
                    target_dir="models",
                    model_name="05_going_modular_tinyvgg_model.pth")






