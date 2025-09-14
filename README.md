# Medical Diagnosis System

A comprehensive Streamlit-based medical diagnosis application with two standalone modules:

1. **Deep Learning (DL)** – Disease diagnosis using CNN + MLP trained on three medical datasets
2. **Intelligent Systems II (IS-II)** – Treatment recommendation using Fuzzy Logic with Genetic Algorithm (GA)

## Features

### Deep Learning Module
- **CNN for Chest X-Ray Analysis**: Convolutional Neural Network for image-based disease detection
- **MLP for Diabetes Classification**: Multi-layer Perceptron for blood test analysis
- **MLP for Symptom Analysis**: Neural network for symptom-based disease prediction
- **Score-Level Fusion**: Weighted combination of predictions from all three models
- **Real-time Training**: Interactive model training with progress visualization
- **Performance Metrics**: Comprehensive evaluation with accuracy, precision, recall, and F1 scores

### Intelligent Systems II Module
- **Fuzzy Logic System**: Patient condition assessment with severity classification
- **Genetic Algorithm Optimization**: Treatment parameter optimization
- **Treatment Database**: Comprehensive treatment options for various diseases
- **Risk Assessment**: Multi-factor risk evaluation
- **Interactive Parameters**: Adjustable GA parameters and risk tolerance

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Final_Project
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## Usage

### Deep Learning Module

1. **Load Datasets**: Click "Load Datasets" to prepare the training data
2. **Train Models**: Click "Train Models" to train CNN and MLP models
3. **Adjust Weights**: Use sidebar sliders to adjust model weights (w_sym, w_blood, w_xray)
4. **Input Patient Data**:
   - Upload chest X-ray image
   - Enter blood test results (age, BMI, blood pressure, glucose, insulin, HbA1c)
   - Select symptoms and patient demographics
5. **Get Diagnosis**: Click "Diagnose" to get weighted fusion results

### Intelligent Systems II Module

1. **Select Disease**: Choose the predicted disease from the Deep Learning module
2. **Input Patient Condition**: Enter vital signs and patient information
3. **Initialize Fuzzy System**: Set up the fuzzy logic system
4. **Fuzzy Analysis**: View severity assessment and risk level
5. **GA Optimization**: Run genetic algorithm with adjustable parameters
6. **View Results**: Get optimized treatment recommendations

## Dataset Requirements

The application is designed to work with three medical datasets:

1. **NIH Chest X-Ray Dataset**: For CNN training
2. **Diabetes Classification Dataset**: For MLP training
3. **Symptom & Patient Profile Dataset**: For symptom-based MLP training

## Architecture

```
app.py                          # Main Streamlit application
├── modules/
│   ├── __init__.py
│   ├── logger.py              # Logging utilities
│   ├── data_handler.py        # Data loading and preprocessing
│   ├── dl_module.py          # Deep Learning module
│   └── is2_module.py         # Intelligent Systems II module
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## Key Features

### Deep Learning Module
- **Modular Architecture**: Separate models for different data types
- **Weighted Fusion**: Configurable weights for model combination
- **Real-time Training**: Interactive model training with progress tracking
- **Comprehensive Evaluation**: Multiple performance metrics
- **Visualization**: Training history and results charts

### Intelligent Systems II Module
- **Fuzzy Logic**: Multi-input severity assessment
- **Genetic Algorithm**: Treatment optimization with customizable parameters
- **Treatment Database**: Comprehensive treatment options
- **Risk Assessment**: Multi-factor risk evaluation
- **Interactive Interface**: Real-time parameter adjustment

## Technical Details

### Deep Learning Models
- **CNN Architecture**: 3-layer convolutional network with max pooling
- **MLP Architecture**: Multi-layer perceptron with dropout
- **Fusion Formula**: Weighted average of model predictions
- **Preprocessing**: Standard scaling and label encoding

### Fuzzy Logic System
- **Input Variables**: Temperature, blood pressure, heart rate, oxygen saturation, age
- **Output Variable**: Severity score (0-1)
- **Membership Functions**: Triangular and trapezoidal functions
- **Rule Base**: Comprehensive rule set for condition assessment

### Genetic Algorithm
- **Population Size**: Configurable (10-100)
- **Generations**: Configurable (10-100)
- **Selection**: Tournament selection
- **Crossover**: Two-point crossover
- **Mutation**: Gaussian mutation
- **Fitness Function**: Multi-objective optimization

## Requirements

- Python 3.8+
- Streamlit 1.28.1
- TensorFlow 2.13.0
- scikit-learn 1.3.0
- scikit-fuzzy 0.4.2
- DEAP 1.3.3
- And other dependencies listed in requirements.txt

## License

This project is for educational and research purposes only. Please ensure compliance with medical data usage regulations and obtain proper permissions for medical dataset usage.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For questions or issues, please contact the development team or create an issue in the repository.
