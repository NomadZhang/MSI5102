# MNIST Presentation Script

Use this file as a slide-by-slide guide. Each section tells you:

- which visual to place on the slide
- what short content should appear on the slide
- what you should say while presenting

---

## Slide 1: Project Overview

**Visual**

- `artifacts/figures/sample_digits.png`

**Put on the slide**

- Goal: classify handwritten digits from the MNIST dataset
- Models compared:
  - k-Nearest Neighbors
  - Logistic Regression
  - Neural Network
- Extra analysis:
  - hyperparameter tuning
  - PCA and t-SNE visualization
  - confusion matrices and metric comparison

**What to say**

- “This project solves handwritten digit recognition on MNIST, which contains 28 by 28 grayscale images of digits from 0 to 9.”
- “Instead of training one model with default settings, we compared three model families and tuned their important hyperparameters.”
- “The goal was not just to get accuracy, but to show a clear methodology, justify parameter choices, and compare strengths and weaknesses.”

---

## Slide 2: Dataset Split and Validation Strategy

**Visual**

- `artifacts/figures/data_split_protocol.png`

**Put on the slide**

- Official MNIST split:
  - 60,000 training images
  - 10,000 test images
- Model-selection split inside the official training set:
  - 51,000 selection-train images
  - 9,000 validation images
- Protocol:
  - use validation split to choose hyperparameters
  - retrain best settings on all 60,000 training images
  - evaluate once on the untouched 10,000-image test set
- We used hold-out validation, not k-fold cross-validation

**What to say**

- “We kept the official 10,000-image test set untouched until the very end, so the final numbers are unbiased.”
- “Inside the 60,000-image training set, we created an 85/15 split: 51,000 images for fitting candidate models and 9,000 images for hyperparameter selection.”
- “We chose hold-out validation instead of k-fold cross-validation because MNIST is already large, the assignment emphasizes implementation and comparison, and repeated k-fold training would add a lot of compute without changing the presentation story much.”
- “After choosing the best parameters, we retrained each winning model on the full 60,000 training images before the final test evaluation.”

---

## Slide 3: Data Exploration and 2D Visualization

**Visual**

- `artifacts/figures/digits_pca_2d.png`
- `artifacts/figures/digits_tsne_2d.png`

**Put on the slide**

- PCA 2D projection explains about **16.9%** of total variance
- PCA shows broad global structure, but some classes still overlap
- t-SNE shows tighter local clusters for many digits
- Digits with similar curved shapes overlap more:
  - 3, 5, 8, 9

**What to say**

- “PCA gives a linear 2D projection, so it is useful for a quick overview of the data structure, but two components only explain about 16.9 percent of the variance.”
- “That means PCA is good for seeing global trends, but not enough to fully separate all digits in two dimensions.”
- “t-SNE gives much clearer local clusters. Digits like 0 and 1 tend to separate well, while curved digits such as 3, 5, 8, and 9 still overlap, which matches the classification errors we later observe.”
- “This directly satisfies the dimensionality-reduction part of the rubric because we are not only plotting the embeddings, we are also interpreting the clustering pattern.”

---

## Slide 4: k-Nearest Neighbors Tuning

**Visual**

- `artifacts/figures/knn_hyperparameter_search.png`

**Put on the slide**

- Hyperparameters explored:
  - `k ∈ {1, 3, 5, 7, 9, 11}`
  - `weights ∈ {uniform, distance}`
- Best validation result:
  - **k = 3**
  - **distance weighting**
  - validation accuracy = **97.089%**
- Final test accuracy = **97.17%**

**What to say**

- “For k-NN, the main parameter is the number of neighbors, and we also compared uniform versus distance weighting.”
- “The best validation performance came from k equals 3 with distance weighting.”
- “This makes sense: very small k can be noisy, while larger k smooths the decision boundary too much and can blur fine digit details.”
- “k-NN achieved strong accuracy, but its tradeoff is inference cost. It has almost no training time, but prediction is much slower because it compares each new image against stored training examples.”

---

## Slide 5: Logistic Regression Tuning and Multiclass Handling

**Visual**

- `artifacts/figures/logistic_regression_hyperparameter_search.png`

**Put on the slide**

- Multiclass strategy:
  - `LogisticRegression` with `lbfgs`
  - for 10 classes, sklearn optimizes **multinomial loss**
- Hyperparameter explored:
  - `C ∈ {0.1, 0.3, 1.0, 3.0}`
- Best validation result:
  - **C = 0.1**
  - validation accuracy = **92.178%**
- Final test accuracy = **92.57%**

**What to say**

- “This is not binary logistic regression repeated manually. With the `lbfgs` solver, sklearn handles the 10-class problem by optimizing multinomial logistic loss.”
- “We tuned the inverse regularization strength C. The best value was 0.1, which means stronger regularization worked better here.”
- “That result is consistent with logistic regression being a linear classifier. It forms a strong baseline, but handwritten digits have nonlinear shape variations, so its ceiling is lower than k-NN and the neural network.”
- “Its weakest classes were digits 8 and 5, which are visually more variable and require more flexible decision boundaries.”

---

## Slide 6: Neural Network Tuning and Chosen Architecture

**Visual**

- `artifacts/figures/neural_network_hyperparameter_search.png`
- `artifacts/figures/neural_network_architecture.png`

**Put on the slide**

- Hyperparameters explored:
  - hidden layers:
    - `(128,)`
    - `(256,)`
    - `(128, 64)`
    - `(256, 128)`
  - `alpha ∈ {1e-4, 5e-4}`
  - learning rate:
    - `1e-3`
    - `5e-4`
- Best validation result:
  - **layers = 256-128**
  - **alpha = 0.0001**
  - **learning rate = 0.001**
  - validation accuracy = **97.700%**
- Final test accuracy = **98.07%**

**What to say**

- “The neural network was the only model family that clearly improved beyond the classical baselines.”
- “We tuned both architecture and training settings instead of using a single arbitrary network.”
- “The best configuration used two hidden layers with 256 and 128 units, a small L2 penalty through alpha, and a learning rate of 0.001.”
- “This architecture is expressive enough to capture nonlinear shape patterns while still training quickly on MNIST.”
- “This slide directly supports the bonus-mark rubric item because it shows both a correct architecture and an evidence-based parameter choice.”

---

## Slide 7: Neural Network Training Evidence

**Visual**

- `artifacts/figures/neural_network_training_curve.png`

**Put on the slide**

- Training loss decreases steadily across epochs
- Validation accuracy stabilizes near the final result
- Early stopping was enabled
- Training setup:
  - Adam optimizer
  - batch size = 256
  - max 35 epochs

**What to say**

- “This figure shows that the network trained in a stable way. The loss keeps going down, and the validation accuracy improves before flattening out.”
- “Early stopping was used so the training does not continue once validation performance stops improving.”
- “That is important because it reduces overfitting risk and supports the claim that the training procedure was reasonable, not just the architecture.”

---

## Slide 8: Final Model Comparison

**Visual**

- `artifacts/figures/model_comparison_dashboard.png`

**Put on the slide**

- Final test metrics:
  - Neural Network: **98.07% accuracy**, **0.98056 macro F1**
  - k-NN: **97.17% accuracy**, **0.97153 macro F1**
  - Logistic Regression: **92.57% accuracy**, **0.92457 macro F1**
- Runtime tradeoff:
  - k-NN has almost zero training time, but slowest prediction
  - Logistic Regression trains fast and is simple
  - Neural Network gives the best accuracy with moderate training cost

**What to say**

- “The neural network is the strongest overall model on this task.”
- “k-NN is competitive, but prediction is much slower because every test image requires distance computations against the training set.”
- “Logistic regression is the weakest on accuracy, but it remains valuable as a clean, interpretable multiclass baseline.”
- “This slide addresses the rubric requirement for clear comparison across models, using multiple metrics and runtime tradeoffs instead of only reporting accuracy.”

---

## Slide 9: Error Analysis and Strengths / Weaknesses

**Visual**

- Main slide:
  - `artifacts/figures/neural_network_confusion_matrix.png`
  - `artifacts/figures/neural_network_confusing_samples.png`
- Appendix or backup visuals:
  - `artifacts/figures/knn_confusing_samples.png`
  - `artifacts/figures/logistic_regression_confusing_samples.png`

**Put on the slide**

- Hardest classes by model:
  - Logistic Regression:
    - digit 8 F1 = **0.883**
    - digit 5 F1 = **0.885**
  - k-NN:
    - digit 9 F1 = **0.959**
  - Neural Network:
    - digit 8 F1 = **0.974**
- Most common wrong label pairs:
  - k-NN:
    - `4 → 9`
    - `7 → 1`
    - `2 → 7`
  - Logistic Regression:
    - `2 → 8`
    - `5 → 3`
    - `4 → 9`
  - Neural Network:
    - `4 → 9`
    - `8 → 3`
    - `5 → 3`
- Main takeaway:
  - confusing curved digits remain the hardest

**What to say**

- “The error pattern matches what we saw in the t-SNE plot. Digits with similar curved shapes are still the hardest to separate.”
- “These misclassified-image galleries are useful because they show why the models are wrong. For example, some 4s look closed at the top and resemble 9s, some 2s have a slanted stroke that makes them resemble 7s, and some 5s and 8s have rounded shapes that overlap with 3.”
- “Logistic regression struggles the most because a linear decision boundary cannot fully model those shape variations.”
- “k-NN handles local patterns better, and the neural network does best because it learns nonlinear internal representations, but even it still makes a small number of shape-based mistakes.”
- “This gives us a more insightful discussion than simply saying one model has the highest score.”

---

## Slide 10: Conclusion and Recommendation

**Visual**

- `artifacts/figures/neural_network_architecture.png`

**Put on the slide**

- Best overall model: **Neural Network**
- Best selected configuration:
  - `256-128` hidden layers
  - `alpha = 0.0001`
  - `learning rate = 0.001`
- Recommended takeaway:
  - Neural Network for highest accuracy
  - k-NN as a strong classical benchmark
  - Logistic Regression as a simple multiclass baseline
- Future work:
  - CNNs
  - cross-validation on a reduced grid
  - more error analysis on similar digits

**What to say**

- “If the goal is best predictive performance, the neural network is the clear recommendation.”
- “If we want a simple nonparametric baseline, k-NN is still strong, but it is much more expensive at prediction time.”
- “If we want interpretability and simplicity, logistic regression remains useful, even though accuracy is lower.”
- “Overall, the project meets the rubric by showing implementation, parameter choice, multiclass handling, dimensionality reduction, comparison, and a clear presentation structure.”

---

## Optional Appendix Slide: Extra Artifacts to Mention if Asked

**Useful files**

- `artifacts/results/model_selection_summary.csv`
- `artifacts/results/model_metrics.csv`
- `artifacts/results/data_split_summary.csv`
- `artifacts/results/knn_hyperparameter_search.csv`
- `artifacts/results/logistic_regression_hyperparameter_search.csv`
- `artifacts/results/neural_network_hyperparameter_search.csv`
- `artifacts/results/knn_top_confusions.csv`
- `artifacts/results/logistic_regression_top_confusions.csv`
- `artifacts/results/neural_network_top_confusions.csv`

**What to say if questioned**

- “These CSV files contain the exact search results and final metrics, so the parameter choice is fully reproducible.”
- “The test set was protected until the end, and the validation split was used only for selecting model settings.”
