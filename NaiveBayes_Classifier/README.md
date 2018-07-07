# Naive Bayes
Naive Bayes methods are a set of supervised learning algorithms based on applying Bayes’ theorem with the “naive” assumption of independence between every pair of features. Given a class variable y and a dependent feature vector x_1 through x_n, Bayes’ theorem states the following relationship:

<img src="http://scikit-learn.org/stable/_images/math/47537fc301bbf8971084f2ecbaa76658ad088235.png" align="middle" />

Using the naive independence assumption that

<img src="http://scikit-learn.org/stable/_images/math/4a2f8ee56238deb56e9e970a749bbe26d96465f7.png" align="middle" />

for all i, this relationship is simplified to

<img src="http://scikit-learn.org/stable/_images/math/7d90ebd2dc0cc2e1136a22625a489c3326d30ec7.png" align="middle" />

Since P(x_1,x_2,..., x_n) is constant given the input, we can use the following classification rule:

<img src="http://scikit-learn.org/stable/_images/math/201f076a3330f2928c26978c4eac59cc8ba4a440.png" align="middle" />

and we can use Maximum A Posteriori (MAP) estimation to estimate P(y) and P(x_i| y); the former is then the relative frequency of class y in the training set.

The different naive Bayes classifiers differ mainly by the assumptions they make regarding the distribution of P(x_i| y).

In spite of their apparently over-simplified assumptions, naive Bayes classifiers have worked quite well in many real-world situations, famously document classification and spam filtering. They require a small amount of training data to estimate the necessary parameters. (For theoretical reasons why naive Bayes works well, and on which types of data it does, see the references below.)

Naive Bayes learners and classifiers can be extremely fast compared to more sophisticated methods. The decoupling of the class conditional feature distributions means that each distribution can be independently estimated as a one dimensional distribution. This in turn helps to alleviate problems stemming from the curse of dimensionality.

On the flip side, although naive Bayes is known as a decent classifier, it is known to be a bad estimator, so the probability outputs from predict_proba are not to be taken too seriously.

## Gaussian Naive Bayes
GaussianNB implements the Gaussian Naive Bayes algorithm for classification. The likelihood of the features is assumed to be Gaussian:

<img src="http://scikit-learn.org/stable/_images/math/ed0c1181c1696f72e1be266187e4694919047d9e.png" align="middle" />

The parameters σ and μ are estimated using maximum likelihood.
## Multinomial Naive Bayes
MultinomialNB implements the naive Bayes algorithm for multinomially distributed data, and is one of the two classic naive Bayes variants used in text classification (where the data are typically represented as word vector counts, although tf-idf vectors are also known to work well in practice). The distribution is parametrized by vectors θ_y = (θ_y1,...θ_yn) for each class y, where n is the number of features (in text classification, the size of the vocabulary) and θ_yi is the probability P(x_i|y) of feature i appearing in a sample belonging to class y.

The parameters θ_y estimated by a smoothed version of maximum likelihood, i.e. relative frequency counting:

<img src="http://scikit-learn.org/stable/_images/math/ac8b939d0915826ce833b43ac13efa8a0fef99d9.png" align="middle" />

where<img src="http://scikit-learn.org/stable/_images/math/54f18d12bb04079aeccdee4418942d30073acc14.png"  />is the number of times feature i appears in a sample of class y in the training set T, and<img src="http://scikit-learn.org/stable/_images/math/125143ac10e186061d40627dae00010cb4eeb04f.png"  />s the total count of all features for class y.

The smoothing priors α>= 0 accounts for features not present in the learning samples and prevents zero probabilities in further computations. Setting α=1 is called Laplace smoothing, while α<1 is called Lidstone smoothing.
## Bernoulli Naive Bayes
BernoulliNB implements the naive Bayes training and classification algorithms for data that is distributed according to multivariate Bernoulli distributions; i.e., there may be multiple features but each one is assumed to be a binary-valued (Bernoulli, boolean) variable. Therefore, this class requires samples to be represented as binary-valued feature vectors; if handed any other kind of data, a BernoulliNB instance may binarize its input (depending on the binarize parameter).

The decision rule for Bernoulli naive Bayes is based on
<img src="http://scikit-learn.org/stable/_images/math/f6a27bc34d0c2ab16cb8ba203e5348741b5521e6.png" align="middle"  />s

which differs from multinomial NB’s rule in that it explicitly penalizes the non-occurrence of a feature i that is an indicator for class y, where the multinomial variant would simply ignore a non-occurring feature.

In the case of text classification, word occurrence vectors (rather than word count vectors) may be used to train and use this classifier. BernoulliNB might perform better on some datasets, especially those with shorter documents. It is advisable to evaluate both models, if time permits.

## Out-of-core naive Bayes model fitting
Naive Bayes models can be used to tackle large scale classification problems for which the full training set might not fit in memory. To handle this case, MultinomialNB, BernoulliNB, and GaussianNB expose a partial_fit method that can be used incrementally as done with other classifiers as demonstrated in Out-of-core classification of text documents. All naive Bayes classifiers support sample weighting.

Contrary to the fit method, the first call to partial_fit needs to be passed the list of all the expected class labels.

For an overview of available strategies in scikit-learn, see also the out-of-core learning documentation.
