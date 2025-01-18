
# CIVIE Omega CNN

This repository contains a pre-trained CNN model for image classification. It allows you to run predictions on single images or perform batch predictions using a benchmark dataset.

## Setup Instructions

### 1. Clone the repository

To clone this repository, run the following command:

```bash
git clone https://github.com/your_username/civie-omega-cnn.git
cd civie-omega-cnn
```

### 2. Create a Virtual Environment

To create a new Python virtual environment, run:

```bash
python3 -m venv temp_env
```

### 3. Install Required Dependencies

After creating and activating the virtual environment, install the required dependencies by running:

```bash
pip install -r requirements.txt
```

### 4. Activate the Virtual Environment

To activate the virtual environment, run:

```bash
source temp_env/bin/activate
```

## Running the Model

### 1. Predict for a Single Image

To run the model on a single image, use the following command:

```bash
python test_single_image.py --image_path "<path_to_your_image>"
```

This command will:
- Load the model from the `model_files` directory.
- Run predictions on the specified image.

### 2. Run Benchmark Prediction

To run a benchmark prediction on a dataset, use the following command:

```bash
python test_benchmark.py
```

This command will:
- Load the model from the `model_files` folder.
- Load the data from the `benchmark/valid.csv` file.
- Store the results in the `benchmark_result` directory.

## Folder Structure

The repository has the following structure:

```
civie-omega-cnn/
│
├── model_files/            # Contains the trained model
├── benchmark/              # Contains the benchmark dataset
│   └── valid.csv           # CSV file with image paths and ground truth
├── benchmark_result/       # Directory where results are saved
├── test_single_image.py    # Script to run prediction on a single image
├── test_benchmark.py       # Script to run benchmark prediction
└── requirements.txt        # Python dependencies
```



