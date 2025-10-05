# Self-Driving Car Simulation — CNN + Udacity Simulator

## Project Overview

This project implements a convolutional neural network (CNN) to predict steering angles for a simulated self-driving car using the Udacity self-driving car simulator. The pipeline covers data collection, preprocessing, model training, and realtime inference while driving in the Udacity simulator.

## Optional Description

A CNN-based behavioral cloning project where the model learns to drive by predicting steering angles from simulator images.

## Features

* CNN model that predicts continuous steering angles from front-facing camera images.
* Data augmentation (flip, brightness, translation, cropping) to improve model robustness.
* Training script with configurable hyperparameters and checkpointing.
* Inference script that streams camera frames to the model and sends steering commands to the Udacity simulator.
* Clear file structure and instructions to reproduce results.

## Requirements

* Python 3.8+ (tested on 3.8/3.9)
* TensorFlow 2.x or Keras (if using standalone Keras adjust imports)
* OpenCV (`opencv-python`)
* NumPy, Pandas
* scikit-learn
* matplotlib (optional, for plotting)
* `udacity-sim` or the official Udacity simulator (install separately) — see below

Install dependencies (example):

```bash
pip install -r requirements.txt
```

`requirements.txt` should include:

```
tensorflow>=2.4
numpy
pandas
opencv-python
matplotlib
scikit-learn
h5py
```

## File structure

```
├── README.md
├── requirements.txt
├── data/
│   ├── driving_log.csv       # CSV with image paths and steering angles
│   ├── IMG/                  # camera images (center,left,right)
│   └── processed/            # optional processed images
├── models/
│   └── checkpoints/          # saved model weights
├── src/
│   ├── data_utils.py         # loading + augmentation + generators
│   ├── model.py              # CNN architecture and model utilities
│   ├── train.py              # training loop and argument parsing
│   ├── drive.py              # realtime inference for simulator
│   └── visualize.py          # plot training history / sample augmentations
├── examples/
│   └── sample_results/      # sample model outputs, loss curves
└── logs/
    └── training_logs/
```

## Dataset and Data Collection

* Use Udacity's sample driving data or collect your own by driving in the simulator and saving images and telemetry to `driving_log.csv`.
* Each row of `driving_log.csv` typically contains: center_image_path, left_image_path, right_image_path, steering_angle, throttle, brake, speed.
* Recommended to record multiple laps and include varied lighting and track conditions.

## Preprocessing and Augmentation

Key steps implemented in `src/data_utils.py`:

* **Crop & Resize:** Crop sky and hood and resize to model input (e.g., 66x200 or 64x160).
* **Color space conversion:** Convert to YUV or YCrCb for better training stability (NVIDIA-style models use YUV).
* **Normalization:** Scale pixel values to range [-0.5, 0.5] or [0,1].
* **Augmentations:** Random horizontal flip (invert steering), brightness jitter, small random translations (adjust steering by a small delta), and random shadowing.
* **Balancing:** Reduce over-representation of zero/near-zero steering angles by downsampling straight-driving frames.
* **Generator:** A Keras-compatible Python generator yields batches of augmented images + angles to avoid loading everything into memory.

## Model Architecture (`src/model.py`)

A commonly used starting architecture (based on NVIDIA model):

* Input: 66x200x3 (YUV) or 64x160x3
* Conv2D(24, 5x5, strides=2) -> ReLU
* Conv2D(36, 5x5, strides=2) -> ReLU
* Conv2D(48, 5x5, strides=2) -> ReLU
* Conv2D(64, 3x3) -> ReLU
* Conv2D(64, 3x3) -> ReLU
* Flatten
* Dense(100) -> Dropout
* Dense(50) -> Dropout
* Dense(10)
* Dense(1)  # output: steering angle (continuous)

Loss: Mean Squared Error (MSE)
Optimizer: Adam (default lr=1e-4)

You can experiment with batch normalization, different dropout rates, and alternative optimizers.

## Training (`src/train.py`)

Usage example:

```bash
python src/train.py --data_csv data/driving_log.csv --batch_size 64 --epochs 10 --save_dir models/checkpoints
```

`train.py` responsibilities:

* Parse command-line args for data paths, hyperparams, model save path.
* Create training/validation split (e.g., 80/20) with `sklearn.model_selection.train_test_split`.
* Instantiate data generators for train/val.
* Build model from `model.py`.
* Compile with MSE loss and callbacks:

  * `ModelCheckpoint` (save best weights)
  * `EarlyStopping` (optional)
  * `CSVLogger` (log training history)
* Plot and save training/validation loss curves to `examples/`.

## Inference & Simulator Integration (`src/drive.py`)

The `drive.py` script connects to the Udacity simulator over HTTP and sends steering/throttle values.

Typical usage (Udacity behavioral cloning style):

```bash
python src/drive.py --model models/checkpoints/best_model.h5
```

`drive.py` responsibilities:

* Launch a small Flask or SocketIO client depending on simulator version (older simulators POST telemetry, newer use socket communication). The repo includes a compatible client for the Udacity simulator provided in the classroom materials.
* On telemetry callback: receive base64-encoded center camera image, decode and preprocess (same pipeline as training), predict steering angle with the trained model.
* Send back a JSON with `steering_angle` and `throttle`. Throttle can be a constant (e.g., 0.2) or controlled by a simple PID controller.
* Optional: save frames and predicted angles for offline analysis.

**Important:** Make sure simulator and `drive.py` protocol versions match. See the Udacity classroom instructions for your simulator's preferred telemetry API.

## Tips for Better Performance

* Use left/right camera images with a correction factor (e.g., +0.2 / -0.2) to teach recovery behavior.
* Heavily augment and randomly translate images so the model learns to recover from off-center positions.
* Reduce the number of near-zero steering examples or add synthetic steering noise to balance the dataset.
* Start with a small learning rate (1e-4) and monitor training/validation loss for overfitting.
* If steering is jittery, apply a small exponential moving average (smoothing) to predictions.

## Evaluation

* Evaluate using validation MSE and, more importantly, by running the car in the simulator.
* Save a video of a successful run using the simulator's built-in recording or by saving frames from `drive.py`.

## Example Commands

Train model:

```bash
python src/train.py --data_csv data/driving_log.csv --epochs 20 --batch_size 64 --save_dir models/checkpoints
```

Drive with saved model:

```bash
python src/drive.py --model models/checkpoints/best_model.h5
```

Visualize training history:

```bash
python src/visualize.py --log logs/training_logs/history.csv
```

## Troubleshooting

* **Simulator disconnects immediately:** check model input shape and dtype; if the model crashes, the server may close the connection.
* **Car steers off abruptly:** lower learning rate, add more augmentation, or apply smoothing on predicted angles.
* **Model predicts constant 0 steering:** dataset is biased (too many straight-driving frames). Rebalance dataset.

## Extensions and Ideas

* Add a temporal component (LSTM or 3-frame stacking) to capture motion.
* Predict throttle and brake along with steering (multi-output network).
* Use transfer learning with a pretrained backbone (MobileNet, ResNet) for better feature extraction.
* Train with higher-resolution images and use more advanced architectures (e.g., attention or residual blocks).

## Credits

* Udacity Self-Driving Car Nanodegree / Simulator telemetry examples.
* NVIDIA's paper on end-to-end learning for self-driving cars (inspiration for the base CNN).

## License

This project is provided under the MIT License — see `LICENSE` for details.

---

If you'd like, I can also:

* Generate `requirements.txt`, starter `train.py`, `model.py`, and `drive.py` templates.
* Create a condensed one-page README or a GitHub-ready repo structure.
