# Water Quality Classification using Neural Networks

This project focuses on building a classification model to predict water potability using a neural network in Keras. The dataset is preprocessed, split into training and testing sets, and then different models are implemented with regularization techniques (L1, L2) to prevent overfitting. The models are evaluated based on their performance using metrics like accuracy, loss, and error analysis, including sensitivity, precision, specificity, and F1 score.

**Link to the Google Colab**
[Water Quality Mode I.ipynb](https://colab.research.google.com/drive/1CEEvAEJ8QbqBHBS-7GkNHd9VcWnKiIrn?usp=sharing#scrollTo=4dowUie-oh8L)

---

## Team Members and Roles

### 1. **Data Handler** - *Esther Mbanzabigwi*
- **Responsibilities**:  
  - Load and preprocess the water quality dataset using Pandas.
  - Split the data into features (X) and labels (Y).
  - Ensure data is properly formatted and scaled.
  - Split the dataset into training (80%) and testing (20%) sets.
  
- **Key Tasks**:  
  - Data loading, preprocessing, and splitting.
  - Collaborating with the model trainers to ensure the dataset is properly formatted for training.

---

### 2. **Vanilla Model Trainer** - *Johnson Tuyishime*
- **Responsibilities**:  
  - Create and train a basic neural network architecture using Keras without regularization.
  - Implement early stopping based on validation loss.

- **Key Tasks**:  
  - Building a standard model with basic layers and activation functions.
  - Sharing the initial findings with the team for comparison with regularized models.

---

### 3. **Model Optimizer 1 (RMSProp, L1 Regularization)** - *Florent Hirwa*
- **Responsibilities**:  
  - Copy the vanilla model and implement L1 regularization.
  - Apply the RMSProp optimizer.
  - Use early stopping to monitor model performance.

- **Key Tasks**:  
  - Focus on L1 regularization and monitor model performance.
  - Compare the performance of the model with the vanilla model.

---

### 4. **Model Optimizer 2 (Adam, L2 Regularization)** - *Maxime Bakunzi*
- **Responsibilities**:  
  - Copy the vanilla model and implement L2 regularization.
  - Apply the Adam optimizer.
  - Use early stopping to monitor model performance.

- **Key Tasks**:  
  - Focus on L2 regularization and monitor model performance.
  - Compare the performance of the model with both the vanilla and L1 regularized models.

- **Additional Responsibility**:  
  - **Error Analysis**: Maxime Bakunzi performed a detailed error analysis using metrics like confusion matrix, sensitivity, precision, specificity, and F1 score.

---

## Project Overview

The project aims to predict water potability based on chemical parameters of water samples. The dataset used in this project contains several water quality measurements that are processed and used to train different neural network models.

### Workflow:

1. **Data Handling**:  
   - Load the dataset.
   - Handle missing values and preprocess the data (scaling and formatting).
   - Split the data into training and testing sets.

2. **Vanilla Model**:  
   - Train a basic neural network without any regularization techniques.
   - Use early stopping to avoid overfitting.

3. **L1 Regularization Model**:  
   - Apply L1 regularization using the RMSProp optimizer.
   - Monitor the model's performance using early stopping.

4. **L2 Regularization Model**:  
   - Apply L2 regularization using the Adam optimizer.
   - Monitor the model's performance using early stopping.

5. **Error Analysis**:  
   - Perform detailed error analysis using metrics like confusion matrix, sensitivity, precision, specificity, and F1 score to understand the performance of the models.

---

## Dataset

- The dataset used for this project contains several water quality parameters and a binary target label (`Potability`).
- The dataset is split into an 80/20 ratio for training and testing purposes.

---

## Model Architecture

All models share a similar neural network architecture but differ in regularization techniques and optimizers:

- **Input Layer**: Dense layer with 64 units and ReLU activation.
- **Hidden Layers**: Two hidden layers with 32 and 16 units respectively, each using ReLU activation.
- **Output Layer**: A single unit with sigmoid activation for binary classification.

---

## Regularization Techniques

1. **Vanilla Model**: No regularization applied, serves as the baseline model.
2. **L1 Regularization**: Uses L1 regularization with RMSProp optimizer to enforce sparsity in the network.
3. **L2 Regularization**: Uses L2 regularization with Adam optimizer to penalize large weights and improve generalization.

---

## Evaluation and Results

Each model was evaluated on the test set, and the performance metrics were calculated. The following table provides a summary of the results:

| Model                  | Accuracy | Loss    | Sensitivity | Precision | Specificity | F1 Score |
|------------------------|----------|---------|-------------|-----------|-------------|----------|
| **Vanilla Model**       | 62.35%   | 0.637   | 0.60        | 0.65      | 0.63        | 0.62     |
| **L1 Regularized Model**| 68.45%   | 0.589   | 0.66        | 0.71      | 0.68        | 0.68     |
| **L2 Regularized Model**| 66.77%   | 0.598   | 0.64        | 0.69      | 0.67        | 0.67     |

---

### Overall Comparison

| **Model**     | **Optimizer** | **Test Loss** | **Test Accuracy** | **Remarks**                                        |
|---------------|---------------|---------------|-------------------|----------------------------------------------------|
| **Vanilla**   | Adam          | 0.8806        | 62.35%            | Baseline, prone to overfitting                     |
| **L1 (RMSProp)** | RMSProp       | 0.6948        | 68.45%            | Best accuracy, prevents overfitting via weight sparsity |
| **L2 (Adam)** | Adam          | 0.6205        | 66.77%            | Better test loss, less overfitting, but slightly lower accuracy than L1 |

---

### Best Performing Model:

L1 Regularization with RMSProp performed the best in terms of accuracy (68.45%), suggesting that the L1 penalty effectively reduces overfitting by creating sparsity in the model's parameters.

## Error Analysis

### Confusion Matrix
A confusion matrix was calculated to provide insights into the performance of the models, showing true positives, false positives, true negatives, and false negatives. 

### Sensitivity, Precision, Specificity, F1 Score
The following metrics were used to evaluate the models' performance in detail:
- **Sensitivity (Recall)**: Measures the model's ability to correctly identify positive instances.
- **Precision**: Reflects the proportion of positive identifications that were actually correct.
- **Specificity**: Measures the model's ability to identify negative cases.
- **F1 Score**: A balance between precision and recall, indicating overall accuracy.

---

## Plots

The following plots were created to visualize model performance:
- **Accuracy Plot**: Training and validation accuracy over epochs.
- **Loss Plot**: Training and validation loss over epochs.

These plots help to compare the vanilla, L1, and L2 models' performance and their ability to generalize on unseen data.

---

## Conclusion

Through this project, we explored how different regularization techniques affect a neural network's performance on the water quality dataset. L1 regularization improved sparsity, while L2 regularization provided more robust models. The Adam and RMSProp optimizers played a significant role in the convergence speed and performance of the models.

---