
"""
K-Detect model configuration.

Central settings for training and inference.
"""

# Reproducibility
RANDOM_STATE = 42

# Dataset settings
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.10

# Traditional ML settings
TFIDF_MAX_FEATURES = 50000
TFIDF_NGRAM_RANGE = (1, 2)
CHAR_NGRAM_RANGE = (3, 5)

# Model settings
LOGISTIC_REGRESSION_MAX_ITER = 1000
SVM_C = 1.0

# Transformer settings
TRANSFORMER_MAX_LENGTH = 256
TRANSFORMER_BATCH_SIZE = 8
TRANSFORMER_LEARNING_RATE = 2e-5
TRANSFORMER_EPOCHS = 3

# Ensemble settings
ENSEMBLE_CALIBRATION_METHOD = "sigmoid"

# Prediction settings
MINIMUM_TEXT_WORDS = 100
