# Machine Learning Glossary (in Plain English)

## Purpose of this document
Machine learning, just like any other discipline, has its fair share of shorthands, acronyms and jargon. This document is meant to give intuitive definitions of ML topics in plain English, in order to help beginners and intermediate ML practitioners alike. Formal definitions can be found elsewhere.

## Contributions
Contributions to this document are more than welcome. If some definition isn't quite correct, or you'd like to add a new one, just submit a PR!

## Glossary

#### Anomaly detection
The process of identifying anomalies from data. Can be treated as an unsupervised learning problem *or* a supervised learning problem, depending on the situation and available data.

#### Autoencoder neural network
A type of neural network that is used for finding efficient encodings, often for the purpose of dimensionality reduction (i.e. reducing the number of features). PCA is another well known algorithm for dimensionality reduction.

Since autoencoder neural networks are trained to reconstruct their own inputs, they are *unsupervised learning algorithms*.

#### Bayes' theorem
The probability that event `A` will occur given that event `B` has occurred.

#### Bayesian probability (subjective probability)
A concept where probabilities are assigned to hypotheses/beliefs; this is contrast to the frequentist way of thinking, wherein a probability is defined as the relative frequency of some occurrence in a large number of trials.

#### Bayesian inference
A from of statistical inference where the probability of a hypothesis (or hypotheses) holding is adjusted as new evidence becomes available.

#### Bagging
See _bootstrap aggregation_.

#### Bootstrap aggregation
An ensemble method to help combat overfitting. First, bootstrapping (see _bootstrap sampling_) is used to make multiple new training sets. Then, multiple models are trained using these data sets. The predictions of each model form the final prediction: for classification, predictions from each model count as "votes", and the majority vote wins; for regression, the predictions from each model are averaged to get the final prediction.

#### Bootstrap sampling
For a a single training set `N`, make `M` new training sets by sampling from the original one. Sampling is done with _replacement_, i.e. once a value is sampled from `N`, it isn't _removed_ from `N`. Thus, the `M` new training sets might (and usually do) share some of the same values.

#### Bucketisation
For a continuous valued variable, the bucketisation is the process of dividing the entire range of values into a set of consecutive _buckets_, each containing a subset of possible values.

#### Classification problems
Problems in which the end goal is to predict which class a data point belongs to. In other words, the output variable follows a discrete distribution (e.g. "small", "medium", "large") as opposed to a continuous distribution (e.g. the set of real numbers `R`).

#### Closed-form expression
An expression that can be evaluated in a finite number of mathematical operations.

#### Collaborative filtering
A widely used supervised learning algorithm for providing recommendations based on user ratings.

#### Convolutional neural network
Informally, a convolutional neural network is a neural network architecture wherein subsets of input neurons are mapped to single neurons in the first hidden layer. Neurons in a given subset share the same weights and biases. Intuitively, convolutional neural networks are particularly good for computer vision problems because they take into account the spacial structure of images by "grouping" nearby pixels together. This is in contrast to simple feed-forward neural networks where image pixels far away from each other are treated the same as pixels near each other (leaving the network to infer spacial structure from training data).

#### Cosine similarity
A measure of similarity between two vectors (the cosine of the angle between them). It's a measure of the similarity of the orientation of two vectors, not their magnitude.

#### Cost function
A function that quantifies the amount by which predictions deviate from observed data/values. Also known as a loss function.

#### Cross-entropy
A loss function typically used in (multinomial) logistic regression / softmax.

#### Deep learning
Using deep neural networks (networks with more than one hidden layer) in order to model higher-level abstractions. A typical example of this is image classification: if we wanted to classify images into pictures of dogs and pictures of cats, a shallow neural network with one hidden layer might detect very primitive shapes and pixel intensity values in order to solve the task. A deep neural network will detect increasingly complex things from one layer to the next, thereby being able to solve more complex problems.

#### Dependent variable
Dependent variables represent the outputs or outcomes under study. Also known as response variables.

#### Ensemble learning
A technique in which multiple machine learning algorithms are combined in order to solve a particular problem.

#### Feed-forward neural network
An artificial neural network in which the connections between neurons do *not* form cycles.

#### F-Score/F-measure
A metric that gives a single numerical value to the combined precision and recall of a query.

#### Gradient descent
An simple algorithm for minimising cost functions. More optimized minimisation algorithms that don't require choosing a learning rate include BFGS and L-BFGS.

#### Independent variable
Independent variables are variables expected to affect the value or one or more dependent variables. Independent variable are also known as predictor variables (predictors).

#### K-means clustering
A popular clustering algorithm used for unsupervised learning tasks (see unsupervised machine learning).

#### Kurtosis
A measure of the "tailedness" of a probability distribution. Positive kurtosis -> sharp peak, negative kurtosis -> small "hump".

#### Linear regression
An approach for modelling the relationship between a dependent variable `y` and `n` independent variables.

#### Logistic regression
A type of regression model where the dependent variable is categorical (e.g. man/woman, malignent/not malignent). Though it's called regression, logistic regression is, in fact, a classification algorithm.

#### Markov chain
A state machine where a transition from A to B happens based on a probability. Typically, Markov chains are memoryless (the probability distribution of possible next states are based solely on the current state). Markov chains where state transitions are based on m past states are called m-order Markov chains (or Markov chains with memory `m`).

#### Markov property
The Markov property is fulfilled iff the process in question is memoryless (see Markov chain).

#### Matthews correlation
A coefficient typically used in binary classification problems as a measure of quality.

#### Multi-layer perceptron
See feed-forward artificial neural networks.

#### Naïve Bayes
A collective term for supervised Bayesian learning algorithms that assume independence between features.

#### Naïve Bayes classifier
A probabilistic classification method/algorithm based on conditional probabilities & Bayes' theorem.

#### (Artificial) Neural Network
A learning algorithm that aims to mimick the function of neurons in the human brain. Can be used for classification and regression problems.

#### One-hot vector
A vector in which all entries are zero except for one with the value 1. Useful for encoding characters, for example. Say we have a vocabulary of four characters [a, b, c, d]. The sequence "c" could be encoded as a one-hot vector [0,0,1,0].

#### One-of-K encoding
See one-hot vector.

#### Overfitting
When a learning example has trained itself "too well" on the training data, making it bad at generalising to new examples (predictions).

#### Perceptron
A feed-forward neural network consisting of two layers (input & output), with only one neuron in the output layer (thus making it a binary classifier). A very simple type of neural network, and one of the first neural networks ever devised.

#### Precision
The amount of true positives returned by a query divided by the amount of examples predicted to be positive.

#### Predictor variable
See independent variable.

#### Principal component analysis (PCA)
The transformation of a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables. Used in machine learning to compress data and reduce dimensionality, thus speeding up training on a data set.

#### Recall
The percentage of true positives returned by a query divided by the total number of true positives.

#### Regression problems
Problems in which the end goal is to predict a continuous valued output for a data point X. This is in contrast to classification problems, where the output can only take on a fixed set of (discrete) values.

#### Regularisation
The process of adding a regularisation term in order to help prevent overfitting models to training data. Used in linear regression, logistic regression, and many other machine learning algorithms.

#### Response variable
See dependent variable.

#### Restriction bias
The set of hypotheses we are restricting ourselves to when solving a machine learning problem using a particular algorithm. For example, if we are using decision trees, we restrict ourselves to only entertaining the set of all possible decision trees, nothing more.

#### Skewness
A measure of the asymmetry of a given probability distribution.

#### Simpson's paradox
The phenomenon where a trend appears in groups of data but disappears/reverses/"cancels out" when the groups of data are aggregated.

#### Softmax
A generalisation of the logistic sigmoid function that "squashes" an vector of real values in to a vector where each value is in the (0,1) range. These values sum up to 1, making softmax good for converting class values into probabilities.

#### Statistical inference
Deducing properties of a population or underlying distribution by analysing data.

#### Supervised machine learning
A field of machine learning wherein algorithms rely on labelled training sets of "correct" examples in order to learn.

#### Support Vector Machine (SVM)
A classifier for classification problems that gives a decision boundary with a "large" margin between classes. Also known as a large margin classifier. SVMs are capable of fitting complex, non-linear hypotheses using the so-called _kernel trick_.

#### Tractable problems
A problem is tractable if, and only if, it can be solved as a closed-form expression (see closed form expression).

#### Unsupervised machine learning
A field of machine learning wherein the training sets are unlabelled and the main objective of learning algorithms is to find structure in the data. A typical example of unsupervised learning is _clustering_.

#### Vectorisation
(In programming) using matrix operations instead of applying operations to individual elements in for/while loops, thereby making computation more efficient.

#### Word embedding
The process in which words from a vocabulary are converted into low-dimensional (low relative to the vocabulary space) vectors of real numbers. Word embeddings can be learned using shallow neural networks.

<br />
<br />
<p align="center"><em>Brought to you, with love, by</em></p>
<p align="center"><a href="https://sc5.io"><img src="https://github.com/SC5/sc5-machine-learning/blob/master/images/sc5logo-small.png" /></a></p>
