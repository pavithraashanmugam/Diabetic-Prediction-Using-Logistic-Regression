## Logistic Regression with Gradient Descent

This project implements **Logistic Regression** using **Gradient Descent** to predict whether a person has diabetes based on several health-related factors. The dataset used is the **Pima Indians Diabetes Database**, which contains multiple features such as glucose levels, age, BMI, and others, with the target variable indicating whether the person has diabetes (1) or not (0).

### Key Concepts

#### 1. **Logistic Regression**:
Logistic regression is a type of regression analysis used for prediction of outcome of a categorical dependent variable based on one or more independent variables. It is particularly used for binary classification problems where the outcome variable is binary (0 or 1).

The logistic regression model uses the **sigmoid function** to predict the probability of the outcome:

\[
y_{\hat} = \frac{1}{1 + e^{-(w \cdot x + b)}}
\]

Where:
- \( y_{\hat} \) is the predicted probability (between 0 and 1).
- \( w \) are the weights associated with the input features \( x \).
- \( b \) is the bias term.

#### 2. **Gradient Descent**:
Gradient descent is an optimization algorithm used to minimize the loss function and update the weights and bias parameters. In this case, it helps minimize the error between the predicted and actual values.

- Weight Update: 
\[
w = w - \alpha \cdot dw
\]
- Bias Update: 
\[
b = b - \alpha \cdot db
\]

Where:
- \( \alpha \) (learning rate) is the step size used to update the parameters.
- \( dw \) and \( db \) are the derivatives of the loss function with respect to weights and bias, respectively.

#### 3. **Loss Function**:
The objective of logistic regression is to minimize the **logistic loss function** (also known as cross-entropy loss), which is given by:

\[
J(w, b) = - \frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(y_{\hat}^{(i)}) + (1 - y^{(i)}) \log(1 - y_{\hat}^{(i)})]
\]

Where:
- \( m \) is the number of training samples.
- \( y^{(i)} \) is the actual outcome (0 or 1) of the \(i^{th}\) sample.
- \( y_{\hat}^{(i)} \) is the predicted probability for the \(i^{th}\) sample.

#### 4. **Learning Rate**:
The learning rate (\(\alpha\)) is a hyperparameter that determines the size of the steps the algorithm takes in the direction of the gradient during the parameter update. Choosing the right learning rate is essential for the algorithm to converge effectively. Too high a learning rate may lead to overshooting the optimal solution, while too low a learning rate can result in a slow convergence.

### Workflow

1. **Data Preprocessing**:
   - The dataset is first loaded and checked for any missing values. In this case, the dataset is clean and does not have missing data.
   - The features are then standardized to ensure all features are within a similar range, which helps gradient descent converge faster.

2. **Train-Test Split**:
   - The dataset is split into training and testing sets using an 80-20 split ratio. This allows us to train the model on one part of the data and test its performance on unseen data.

3. **Model Training**:
   - We initialize weights and biases as zero and then use gradient descent to optimize these parameters over a fixed number of iterations (1000 in this case).

4. **Model Evaluation**:
   - The trained model is evaluated on both the training and testing datasets to compute the accuracy of predictions.
   - The accuracy is computed as the percentage of correct predictions.

### Model Results

- **Training Data Accuracy**: 77.69%
- **Testing Data Accuracy**: 76.62%

These accuracies indicate that the model is performing reasonably well, but there may still be room for improvement. Further optimization of hyperparameters (e.g., learning rate, number of iterations) or using a different optimization technique (e.g., stochastic gradient descent, or using a more advanced model like decision trees or neural networks) could lead to better results.

### Predictive System

To use the model for prediction:
1. The model takes in a set of feature values.
2. The features are standardized using the same scaler fitted on the training data.
3. The model predicts whether the person is diabetic (1) or not (0) based on the learned weights and bias.

For example, the input data `(5, 0, 0, 0, 175, 0.8, 0.587, 51)` returns the prediction `0`, indicating that the person is not diabetic.

### Code Overview

The code defines a `Logistic_Regression` class with methods:
- `fit()`: Trains the model by running gradient descent to update the weights and bias.
- `update_weights()`: Implements the gradient descent update rule.
- `predict()`: Uses the sigmoid function to make predictions based on input features.

### Libraries Used

- **NumPy**: For handling numerical computations, including matrix operations and vectorized implementations of mathematical functions.
- **Pandas**: For loading and manipulating the dataset.
- **Scikit-learn**: For preprocessing (StandardScaler) and model evaluation (train-test split and accuracy score).
- **Matplotlib**: (Optional, not included in the code above) for visualization.

### Conclusion

This simple logistic regression implementation demonstrates the process of building a binary classification model using gradient descent. Despite being a basic model, it achieves decent performance on the diabetes dataset and can be further improved with advanced techniques.
