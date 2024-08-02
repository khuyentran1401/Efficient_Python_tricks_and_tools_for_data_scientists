## Machine Learning

![](../img/machine_learning.png)

### causalimpact: Find Causal Relation of an Event and a Variable in Python	


```python
!pip install pycausalimpact
```

When working with time series data, you might want to determine whether an event has an impact on some response variable or not. For example, if your company creates an advertisement, you might want to track whether the advertisement results in an increase in sales or not.

That is when causalimpact comes in handy. causalimpact analyses the differences between expected and observed time series data. With causalimpact, you can infer the expected effect of an intervention in 3 lines of code.


```python
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import ArmaProcess
import causalimpact
from causalimpact import CausalImpact

# Generate random sample

np.random.seed(0)
ar = np.r_[1, 0.9]
ma = np.array([1])
arma_process = ArmaProcess(ar, ma)

X = 50 + arma_process.generate_sample(nsample=1000)
y = 1.6 * X + np.random.normal(size=1000)

# There is a change starting from index 800
y[800:] += 10
```


```python
data = pd.DataFrame({"y": y, "X": X}, columns=["y", "X"])
pre_period = [0, 799]
post_period = [800, 999]
```


```python
ci = CausalImpact(data, pre_period, post_period)
print(ci.summary())
print(ci.summary(output="report"))
ci.plot()
```

    Posterior Inference {Causal Impact}
                              Average            Cumulative
    Actual                    90.03              18006.16
    Prediction (s.d.)         79.97 (0.3)        15994.43 (60.93)
    95% CI                    [79.36, 80.55]     [15871.15, 16110.0]
    
    Absolute effect (s.d.)    10.06 (0.3)        2011.72 (60.93)
    95% CI                    [9.48, 10.68]      [1896.16, 2135.01]
    
    Relative effect (s.d.)    12.58% (0.38%)     12.58% (0.38%)
    95% CI                    [11.86%, 13.35%]   [11.86%, 13.35%]
    
    Posterior tail-area probability p: 0.0
    Posterior prob. of a causal effect: 100.0%
    
    For more details run the command: print(impact.summary('report'))
    Analysis report {CausalImpact}
    
    
    During the post-intervention period, the response variable had
    an average value of approx. 90.03. By contrast, in the absence of an
    intervention, we would have expected an average response of 79.97.
    The 95% interval of this counterfactual prediction is [79.36, 80.55].
    Subtracting this prediction from the observed response yields
    an estimate of the causal effect the intervention had on the
    response variable. This effect is 10.06 with a 95% interval of
    [9.48, 10.68]. For a discussion of the significance of this effect,
    see below.
    
    
    Summing up the individual data points during the post-intervention
    period (which can only sometimes be meaningfully interpreted), the
    response variable had an overall value of 18006.16.
    By contrast, had the intervention not taken place, we would have expected
    a sum of 15994.43. The 95% interval of this prediction is [15871.15, 16110.0].
    
    
    The above results are given in terms of absolute numbers. In relative
    terms, the response variable showed an increase of +12.58%. The 95%
    interval of this percentage is [11.86%, 13.35%].
    
    
    This means that the positive effect observed during the intervention
    period is statistically significant and unlikely to be due to random
    fluctuations. It should be noted, however, that the question of whether
    this increase also bears substantive significance can only be answered
    by comparing the absolute effect (10.06) to the original goal
    of the underlying intervention.
    
    
    The probability of obtaining this effect by chance is very small
    (Bayesian one-sided tail-area probability p = 0.0).
    This means the causal effect can be considered statistically
    significant.



    
![png](machine_learning_files/machine_learning_7_1.png)
    


### Scikit-LLM: Supercharge Text Analysis with ChatGPT and scikit-learn Integration 


```python
!pip install scikit-llm
```

To integrate advanced language models with scikit-learn for enhanced text analysis tasks, use Scikit-LLM.

Scikit-LLM's `ZeroShotGPTClassifier` enables text classification on unseen classes without requiring re-training. 


```python
from skllm.config import SKLLMConfig

SKLLMConfig.set_openai_key("<YOUR_KEY>")
SKLLMConfig.set_openai_org("<YOUR_ORGANISATION>")
```


```python
from skllm.datasets import get_classification_dataset
from skllm import ZeroShotGPTClassifier

# demo sentiment analysis dataset
# labels: positive, negative, neutral
X, y = get_classification_dataset()

clf = ZeroShotGPTClassifier(openai_model="gpt-3.5-turbo")
clf.fit(X, y)
labels = clf.predict(X)
```

    100%|██████████| 30/30 [00:36<00:00,  1.22s/it]



```python
from sklearn.metrics import accuracy_score
print(f"Accuracy: {accuracy_score(y, labels):.2f}")
```

    Accuracy: 0.93


[Link to Scikit-LLM](https://github.com/iryna-kondr/scikit-llm).

### Create a Readable Machine Learning Pipeline in One Line of Code

If you want to create a readable machine learning pipeline in a single line of code, try the `make_pipeline` function in scikit-learn.
`make_pipeline` is especially useful when working with complex pipelines that involve many different transformers and estimators.


```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Create a pipeline that scales the data and fits a logistic regression model
pipeline = make_pipeline(StandardScaler(), LogisticRegression())

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Evaluate the pipeline on the test data
pipeline.score(X_test, y_test)

```




    0.96



### Pipeline + GridSearchCV: Prevent Data Leakage when Scaling the Data

Scaling the data before using GridSearchCV can lead to data leakage since the scaling tells some information about the entire data. To prevent this, assemble both the scaler and machine learning models in a pipeline then use it as the estimator for GridSearchCV. 


```python
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.datasets import load_iris

# load data
df = load_iris()
X = df.data
y = df.target

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Create a pipeline variable
make_pipe = make_pipeline(StandardScaler(), SVC())

# Defining parameters grid
grid_params = {"svc__C": [0.1, 1, 10, 100, 1000], "svc__gamma": [0.1, 1, 10, 100]}

# hypertuning
grid = GridSearchCV(make_pipe, grid_params, cv=5)
grid.fit(X_train, y_train)

# predict
y_pred = grid.predict(X_test)
```


    <IPython.core.display.Javascript object>


The estimator is now the entire pipeline instead of just the machine learning model.

### squared=False: Get RMSE from Sklearn’s mean_squared_error method

If you want to get the root mean squared error using sklearn, pass `squared=False` to sklearn’s `mean_squared_error` method.


```python
from sklearn.metrics import mean_squared_error

y_actual = [1, 2, 3]
y_predicted = [1.5, 2.5, 3.5]
rmse = mean_squared_error(y_actual, y_predicted, squared=False)
rmse
```




    0.5




    <IPython.core.display.Javascript object>


### modelkit: Build Production ML Systems in Python


```python
!pip install modelkit textblob 
```

If you want your ML models to be fast, type-safe, testable, and fast to deploy to production, try modelkit. modelkit allows you to incorporate all of these features into your model in several lines of code. 


```python
from modelkit import ModelLibrary, Model 
from textblob import TextBlob, WordList
# import nltk
# nltk.download('brown')
# nltk.download('punkt')
```

To define a modelkit Model, you need to:

- create class inheriting from `modelkit.Model`
- implement a `_predict` method


```python
class NounPhraseExtractor(Model):

    # Give model a name
    CONFIGURATIONS = {"noun_phrase_extractor": {}}

    def _predict(self, text):
        blob = TextBlob(text)
        return blob.noun_phrases
```


    <IPython.core.display.Javascript object>


You can now instantiate and use the model:


```python
noun_extractor = NounPhraseExtractor()
noun_extractor("What are your learning strategies?")
```

    2021-11-05 09:55.55 [debug    ] Model loaded                   memory=0 Bytes memory_bytes=0 model_name=None time=0 microseconds time_s=4.232699939166196e-05





    WordList(['learning strategies'])




    <IPython.core.display.Javascript object>


You can also create test cases for your model and make sure all test cases are passed. 


```python
class NounPhraseExtractor(Model):

    # Give model a name
    CONFIGURATIONS = {"noun_phrase_extractor": {}}

    TEST_CASES = [
        {"item": "There is a red apple on the tree", "result": WordList(["red apple"])}
    ]

    def _predict(self, text):
        blob = TextBlob(text)
        return blob.noun_phrases
```


    <IPython.core.display.Javascript object>



```python
noun_extractor = NounPhraseExtractor()
noun_extractor.test()
```

    2021-11-05 09:55.58 [debug    ] Model loaded                   memory=0 Bytes memory_bytes=0 model_name=None time=0 microseconds time_s=4.3191997974645346e-05



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008000; text-decoration-color: #008000">TEST </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">1</span><span style="color: #008000; text-decoration-color: #008000">: SUCCESS</span>
</pre>




    <IPython.core.display.Javascript object>


modelkit also allows you to organize a group of models using `ModelLibrary`.


```python
class SentimentAnalyzer(Model):
    
    # Give model a name
    CONFIGURATIONS = {"sentiment_analyzer": {}}

    def _predict(self, text):
        blob = TextBlob(text)
        return blob.sentiment
```


```python
nlp_models = ModelLibrary(models=[NounPhraseExtractor, SentimentAnalyzer])
```

    2021-11-05 09:50.13 [info     ] Instantiating AssetsManager    lazy_loading=False
    2021-11-05 09:50.13 [info     ] No remote storage provider configured
    2021-11-05 09:50.13 [debug    ] Resolving asset for Model      model_name=sentiment_analyzer
    2021-11-05 09:50.13 [debug    ] Loading model                  model_name=sentiment_analyzer
    2021-11-05 09:50.13 [debug    ] Instantiating Model object     model_name=sentiment_analyzer
    2021-11-05 09:50.13 [debug    ] Model loaded                   memory=0 Bytes memory_bytes=0 model_name=sentiment_analyzer time=0 microseconds time_s=3.988200114690699e-05
    2021-11-05 09:50.13 [debug    ] Done loading Model             model_name=sentiment_analyzer
    2021-11-05 09:50.13 [info     ] Model and dependencies loaded  memory=0 Bytes memory_bytes=0 name=sentiment_analyzer time=0 microseconds time_s=0.00894871700074873
    2021-11-05 09:50.13 [debug    ] Resolving asset for Model      model_name=noun_phrase_extractor
    2021-11-05 09:50.13 [debug    ] Loading model                  model_name=noun_phrase_extractor
    2021-11-05 09:50.13 [debug    ] Instantiating Model object     model_name=noun_phrase_extractor
    2021-11-05 09:50.13 [debug    ] Model loaded                   memory=0 Bytes memory_bytes=0 model_name=noun_phrase_extractor time=0 microseconds time_s=2.751099964370951e-05
    2021-11-05 09:50.13 [debug    ] Done loading Model             model_name=noun_phrase_extractor
    2021-11-05 09:50.13 [info     ] Model and dependencies loaded  memory=0 Bytes memory_bytes=0 name=noun_phrase_extractor time=0 microseconds time_s=0.006440052002290031


Get and use the models from `nlp_models`.


```python
noun_extractor = model_collections.get("noun_phrase_extractor")
noun_extractor("What are your learning strategies?")
```




    WordList(['learning strategies'])




```python
sentiment_analyzer = model_collections.get("sentiment_analyzer")
sentiment_analyzer("Today is a beautiful day!")
```




    Sentiment(polarity=1.0, subjectivity=1.0)



[Link to modelkit](https://github.com/Cornerstone-OnDemand/modelkit/).

### Decompose high dimensional data into two or three dimensions


```python
!pip install yellowbrick
```

If you want to decompose high dimensional data into two or three dimensions to visualize it, what should you do? A common technique is PCA. Even though PCA is useful, it can be complicated to create a PCA plot.

Lucikily, Yellowbrick allows you visualize PCA in a few lines of code


```python
from yellowbrick.datasets import load_credit
from yellowbrick.features import PCA
```


    <IPython.core.display.Javascript object>



```python
X, y = load_credit()
classes = ["account in defaut", "current with bills"]
```


    <IPython.core.display.Javascript object>



```python
visualizer = PCA(scale=True, classes=classes)
visualizer.fit_transform(X, y)
visualizer.show()
```


    
![png](machine_learning_files/machine_learning_48_0.png)
    





    <AxesSubplot:title={'center':'Principal Component Plot'}, xlabel='$PC_1$', ylabel='$PC_2$'>




    <IPython.core.display.Javascript object>


[Link to Yellowbrick](https://www.scikit-yb.org/en/latest/)

### Visualize Feature Importances with Yellowbrick


```python
!pip install yellowbrick
```

Having more features is not always equivalent to a better model. The more features a model has, the more sensitive the model is to errors due to variance. Thus, we want to select the minimum required features to produce a valid model.

A common approach to eliminate features is to eliminate the ones that are the least important to the model. Then we re-evaluate if the model actually performs better during cross-validation.

Yellowbrick's `FeatureImportances` is ideal for this task since it helps us to visualize the relative importance of the features for the model.


```python
from sklearn.tree import DecisionTreeClassifier
from yellowbrick.datasets import load_occupancy
from yellowbrick.model_selection import FeatureImportances
```


```python
X, y = load_occupancy()

model = DecisionTreeClassifier()

viz = FeatureImportances(model)
viz.fit(X, y)
viz.show();
```


    
![png](machine_learning_files/machine_learning_54_0.png)
    


From the plot above, it seems like the light is the most important feature to DecisionTreeClassifier, followed by CO2, temperature.


[Link to Yellowbrick](https://www.scikit-yb.org/en/latest/)

### Validation Curve: Determine if an Estimator Is Underfitting Over Overfitting


```python
!pip install yellowbrick
```

To find the hyperparameter where the estimator is neither underfitting nor overfitting, use Yellowbrick's validation curve.


```python
from yellowbrick.datasets.loaders import load_occupancy
from yellowbrick.model_selection import validation_curve
from sklearn.tree import DecisionTreeClassifier

import numpy as np 
```


```python
# Load data
X, y = load_occupancy()
```

In the code below, we choose the range of `max_depth` to be from 1 to 11.


```python
viz = validation_curve(
    DecisionTreeClassifier(), X, y, param_name="max_depth",
    param_range=np.arange(1, 11), cv=10, scoring="f1",
)
```


    
![png](machine_learning_files/machine_learning_63_0.png)
    


As we can see from the plot above, although `max_depth` > 2 has a higher training score but a lower cross-validation score. This indicates that the model is overfitting. 

Thus, the sweet spot will be where the cross-validation score neither increases nor decreases, which is 2.

[Link to Yellowbrick](https://www.scikit-yb.org/en/latest/index.html).

[My full article about Yellowbrick](https://towardsdatascience.com/introduction-to-yellowbrick-a-python-library-to-explain-the-prediction-of-your-machine-learning-d63ecee10ecc).

### Mlxtend: Plot Decision Regions of Your ML Classifiers


```python
!pip install mlxtend  
```

How does your machine learning classifier decide which class a sample belongs to? Plotting a decision region can give you some insights into your ML classifier's decision.

An easy way to plot decision regions is to use mlxtend's `plot_decision_regions`.


```python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import EnsembleVoteClassifier
from mlxtend.data import iris_data
from mlxtend.plotting import plot_decision_regions

# Initializing Classifiers
clf1 = LogisticRegression(random_state=0)
clf2 = RandomForestClassifier(random_state=0)
clf3 = SVC(random_state=0, probability=True)
eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], weights=[2, 1, 1], voting='soft')

# Loading some example data
X, y = iris_data()
X = X[:,[0, 2]]

# Plotting Decision Regions
gs = gridspec.GridSpec(2, 2)
fig = plt.figure(figsize=(10, 8))

for clf, lab, grd in zip([clf1, clf2, clf3, eclf],
                         ['Logistic Regression', 'Random Forest', 'RBF kernel SVM', 'Ensemble'],
                         itertools.product([0, 1], repeat=2)):
    clf.fit(X, y)
    ax = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=X, y=y, clf=clf, legend=2)
    plt.title(lab)
plt.show()
```


    
![png](machine_learning_files/machine_learning_69_0.png)
    


Mlxtend (machine learning extensions) is a Python library of useful tools for the day-to-day data science tasks. 

Find other useful functionalities of Mlxtend [here](https://github.com/rasbt/mlxtend).

### Deepchecks + Weights & Biases: Test and Track Your ML Model and Data


```python
!pip install -U deepchecks wandb scikit-learn 
```

[Weight and Biases](https://towardsdatascience.com/introduction-to-weight-biases-track-and-visualize-your-machine-learning-experiments-in-3-lines-9c9553b0f99d) is a tool to track and monitor your ML experiments. [deepchecks](https://deepchecks.com/) is a tool that allows you to create test suites for your ML models & data with ease. 

The checks in a suite includes:
- model performance
- data integrity
- distribution mismatches
and more.

Now you can track deepchecks suite's results with Weights & Biases. 

To learn how to use this integration, start with loading a dataset and splitting it into train and test set.




```python
import pandas as pd  
import numpy as np 
from sklearn.datasets import load_iris 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from deepchecks import Dataset

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
```


```python
# Load Data
X, y = load_iris(as_frame=True, return_X_y=True)
label_col = 'target'

df = pd.concat([X, y.to_frame(name=label_col)], axis=1)
```


```python
df_train, df_test = train_test_split(df, stratify=df[label_col], random_state=0)
```

Next, build a ML model using the training data. 


```python
rf_clf = RandomForestClassifier()
rf_clf.fit(df_train.drop(label_col, axis=1), df_train[label_col])
```




    RandomForestClassifier()



Create deepchecks' Dataset objects with train and test set. 


```python
ds_train = Dataset(df_train, label=label_col, cat_features=[])
ds_test =  Dataset(df_test,  label=label_col, cat_features=[])
```

Create a test suite using our ML model and datasets and run it.


```python
from deepchecks.suites import full_suite

suite = full_suite()
suite_result = suite.run(ds_train, ds_test, rf_clf)
```


    Full Suite:   0%|          | 0/36 [00:00<?, ? Check/s]


Export all results to Weights & Biases:


```python
import wandb
wandb.login()
```

    [34m[1mwandb[0m: Currently logged in as: [33mkhuyentran1401[0m (use `wandb login --relogin` to force relogin)





    True




```python
suite_result.to_wandb() 
```


Tracking run with wandb version 0.12.11



Run data is saved locally in <code>/home/khuyen/book/book/Chapter5/wandb/run-20220314_094658-1yf63l3g</code>



Syncing run <strong><a href="https://wandb.ai/khuyentran1401/deepchecks/runs/1yf63l3g" target="_blank">mud-mousse-2</a></strong> to <a href="https://wandb.ai/khuyentran1401/deepchecks" target="_blank">Weights & Biases</a> (<a href="https://wandb.me/run" target="_blank">docs</a>)<br/>



    Full Suite:   0%|          | 0/52 [00:00<?, ? Check/s]


    



Waiting for W&B process to finish... <strong style="color:green">(success).</strong>



    VBox(children=(Label(value='0.322 MB of 0.322 MB uploaded (0.000 MB deduped)\r'), FloatProgress(value=0.999393…



Synced <strong style="color:#cdcd00">mud-mousse-2</strong>: <a href="https://wandb.ai/khuyentran1401/deepchecks/runs/1yf63l3g" target="_blank">https://wandb.ai/khuyentran1401/deepchecks/runs/1yf63l3g</a><br/>Synced 4 W&B file(s), 109 media file(s), 90 artifact file(s) and 0 other file(s)



Find logs at: <code>./wandb/run-20220314_094658-1yf63l3g/logs</code>


And this is how the test suite will look like in W&B.

![](../img/deepchecks_wandb.gif)

[My article about Weights & Biases](https://towardsdatascience.com/introduction-to-weight-biases-track-and-visualize-your-machine-learning-experiments-in-3-lines-9c9553b0f99d)

[Link to deepchecks](https://deepchecks.com/)

### imbalanced-learn: Deal with an Imbalanced Dataset


```python
!pip install imbalanced-learn==0.10.0 mlxtend==0.21.0
```

A dataset is imbalanced when the number of samples in one class is much more than the other classes. When training a sensitive classifier using an imbalanced dataset, it will work well on the majority class but work poorly on the minority class.

To deal with an imbalanced dataset, we can use imbalanced-learn to generate new samples in the classes which are under-represented. 

In the example below, we use the `RandomOverSampler` class from imbalanced-learn to generate new samples by randomly sampling with replacement the current available samples.


```python
# Libraries for plotting
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.plotting import plot_decision_regions
import matplotlib.gridspec as gridspec

# Libraries for machine learning
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC
import warnings

warnings.simplefilter("ignore", UserWarning)

```


```python
X, y = make_classification(
    n_samples=5000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    n_classes=4,
    n_clusters_per_class=1,
    weights=[0.01, 0.04, 0.5, 0.90],
    class_sep=0.8,
    random_state=0,
)

```


```python
from imblearn.over_sampling import RandomOverSampler


ros = RandomOverSampler(random_state=1)
X_resampled, y_resampled = ros.fit_resample(X, y)

```


```python
# Plotting Decision Regions
fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharey=True, figsize=(6, 10))

for Xi, yi, ax, title in zip(
    [X, X_resampled],
    [y, y_resampled],
    [ax0, ax1],
    ["Without resampling", "Using RandomOverSampler"],
):
    clf = LinearSVC()
    clf.fit(Xi, yi)
    fig = plot_decision_regions(X=Xi, y=yi, clf=clf, legend=2, ax=ax, colors='#A3D9B1,#06B1CF,#F8D347,#E48789')
    plt.title(title)

```


    
![png](machine_learning_files/machine_learning_95_0.png)
    


[Link to imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn).

### Estimate Prediction Intervals in Scikit-Learn Models with MAPIE 


```python
!pip install mapie
```

To get estimated prediction intervals for predictions made by a scikit-learn model, use MAPIE.

In the code below, we use `MapieRegressor` to estimate prediction intervals for a scikit-learn regressor. 


```python
from mapie.regression import MapieRegressor
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

# Create data
X, y = make_regression(n_samples=200, n_features=1, noise=50, random_state=0)

# Train and predict
alpha = [0.05, 0.32]
mapie = MapieRegressor(LinearRegression())
mapie.fit(X, y)
y_pred, y_pis = mapie.predict(X, alpha=alpha)

```


```python
# compute the coverage of the prediction intervals
from mapie.metrics import regression_coverage_score

coverage_scores = [
    regression_coverage_score(y, y_pis[:, 0, i], y_pis[:, 1, i])
    for i, _ in enumerate(alpha)
]

```


```python
# Plot the estimated prediction intervals

from matplotlib import pyplot as plt
import numpy as np

plt.figure(figsize=(11, 7))
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20) 
plt.scatter(X, y, alpha=0.3, c='#06B1CF', s=80)
plt.plot(X, y_pred, color="#E48789", linewidth=3)
order = np.argsort(X[:, 0])
plt.plot(X[order], y_pis[order][:, 0, 1], color="#E48789", ls="--", linewidth=3)
plt.plot(X[order], y_pis[order][:, 1, 1], color="#E48789", ls="--", linewidth=3)
plt.fill_between(
    X[order].ravel(),
    y_pis[order][:, 0, 0].ravel(),
    y_pis[order][:, 1, 0].ravel(),
    alpha=0.2,
    color="#E48789"
)
plt.title(
    f"Target coverages for "
    f"alpha={alpha[0]:.2f}: ({1-alpha[0]:.3f}, {coverage_scores[0]:.3f})\n"
    f"Target coverages for "
    f"alpha={alpha[1]:.2f}: ({1-alpha[1]:.3f}, {coverage_scores[1]:.3f})",
    fontsize=20,
)
plt.show()

```


    
![png](machine_learning_files/machine_learning_103_0.png)
    


[Link to MAPIE](https://github.com/scikit-learn-contrib/MAPIE).

### mlforecast: Scalable Machine Learning for Time Series

If you want to perform time series forecasting using machine learning models and scale to massive amounts of data with distributed training, try mlforecast. 

```python
from mlforecast.distributed import DistributedMLForecast
from mlforecast.distributed.models.dask.lgb import DaskLGBMForecast
from mlforecast.target_transforms import Differences

# Create Dask Dataframe
series_ddf = ...

# Perform distributed training
fcst = DistributedMLForecast(
    models=DaskLGBMForecast(),
    freq='D', # daily frequency
    lags=[7], 
    target_transforms=[Differences([1])],
)
fcst.fit(series_ddf)
```

[Full code of the example above](https://nixtla.github.io/mlforecast/docs/quick_start_distributed.html).
[Link to mlforecast](https://github.com/Nixtla/mlforecast).

### MLEM: Capture Your Machine Learning Model's Metadata


```python
!pip install mlem 
```

The metadata of a machine learning model provides important information about the model such as:

- Hash value
- Model methods
- Input data schema
- Python requirements used to train the model. 

This information enables others to reproduce the model and its results. 

With MLEM, you can save both the model and its metadata in a single line of code.


```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=0)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training set
model.fit(X_train, y_train)
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LinearRegression</label><div class="sk-toggleable__content"><pre>LinearRegression()</pre></div></div></div></div></div>




```python
from mlem.api import save 

# Instead of joblib.dump(model, 'model/diabetes_model')
save(model, 'model/diabetes_model', sample_data=X_test)
```




    MlemModel(location=Location(path='/Users/khuyentran/book/Efficient_Python_tricks_and_tools_for_data_scientists/Chapter5/model/diabetes_model.mlem', project=None, rev=None, uri='file:///Users/khuyentran/book/Efficient_Python_tricks_and_tools_for_data_scientists/Chapter5/model/diabetes_model.mlem', project_uri=None, fs=<fsspec.implementations.local.LocalFileSystem object at 0x16b631430>), params={}, artifacts={'data': LocalArtifact(uri='diabetes_model', size=563, hash='c57e456e8a0768326655a8b52cde4f47')}, requirements=Requirements(__root__=[InstallableRequirement(module='sklearn', version='1.2.1', package_name='scikit-learn', extra_index=None, source_url=None, vcs=None, vcs_commit=None), InstallableRequirement(module='numpy', version='1.24.2', package_name=None, extra_index=None, source_url=None, vcs=None, vcs_commit=None)]), processors_cache={'model': SklearnModel(model=LinearRegression(), io=SimplePickleIO(), methods={'predict': Signature(name='predict', args=[Argument(name='X', type_=NumpyNdarrayType(value=None, shape=(None, 10), dtype='float64'), required=True, default=None, kw_only=False)], returns=NumpyNdarrayType(value=None, shape=(None,), dtype='float64'), varargs=None, varargs_type=None, varkw=None, varkw_type=None)})}, call_orders={'predict': [('model', 'predict')]})



Running the code above will create two files: a model file and a metadata file.

```bash
model
 ├──  diabetes_model  
 └──  diabetes_model.mlem  
```

Here is what the metadata file looks like:

```yaml
# model/diabetes_model.mlem

artifacts:
  data:
    hash: c57e456e8a0768326655a8b52cde4f47
    size: 563
    uri: diabetes_model
call_orders:
  predict:
  - - model
    - predict
object_type: model
processors:
  model:
    methods:
      predict:
        args:
        - name: X
          type_:
            dtype: float64
            shape:
            - null
            - 10
            type: ndarray
        name: predict
        returns:
          dtype: float64
          shape:
          - null
          type: ndarray
    type: sklearn
requirements:
- module: sklearn
  package_name: scikit-learn
  version: 1.2.1
- module: numpy
  version: 1.24.2
```

[Link to MLEM](https://mlem.ai/).

[How to deploy your model with MLEM](https://towardsdatascience.com/automate-machine-learning-deployment-with-github-actions-f752766981b1?sk=6e234bb505e6bc426bb4760e4b20da1b).

### Distributed Machine Learning with MLlib


```python
!pip install pyspark
```

If you want to perform distributed machine learning tasks and handle large-scale datasets, use MLlib. It's designed to work seamlessly with Apache Spark, making it a powerful tool for scalable machine learning.

Similar to scikit-learn, MLlib provides the following tools:
- ML Algorithms: Classification, regression, clustering, and collaborative filtering
- Featurization: Feature extraction, transformation, dimensionality reduction, and selection
- Pipelines: Construction, evaluation, and tuning of ML Pipelines


```python
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# Prepare training data from a list of (label, features) tuples.
training = spark.createDataFrame(
    [
        (1.0, Vectors.dense([0.0, 1.1, 0.1])),
        (0.0, Vectors.dense([2.0, 1.0, -1.0])),
        (0.0, Vectors.dense([2.0, 1.3, 1.0])),
        (1.0, Vectors.dense([0.0, 1.2, -0.5])),
    ],
    ["label", "features"],
)

# Prepare test data
test = spark.createDataFrame(
    [
        (1.0, Vectors.dense([-1.0, 1.5, 1.3])),
        (0.0, Vectors.dense([3.0, 2.0, -0.1])),
        (1.0, Vectors.dense([0.0, 2.2, -1.5])),
    ],
    ["label", "features"],
)

# Create a LogisticRegression instance. This instance is an Estimator.
lr = LogisticRegression(maxIter=10, regParam=0.01)

# Learn a LogisticRegression model. This uses the parameters stored in lr.
model1 = lr.fit(training)

# Make predictions on test data using the Transformer.transform() method.
# LogisticRegression.transform will only use the 'features' column.
prediction = model1.transform(test)
result = prediction.select("features", "label", "probability", "prediction").collect()

for row in result:
    print(
        "features=%s, label=%s -> prob=%s, prediction=%s"
        % (row.features, row.label, row.probability, row.prediction)
    )

```

    LogisticRegression parameters:
    aggregationDepth: suggested depth for treeAggregate (>= 2). (default: 2)
    elasticNetParam: the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty. (default: 0.0)
    family: The name of family which is a description of the label distribution to be used in the model. Supported options: auto, binomial, multinomial (default: auto)
    featuresCol: features column name. (default: features)
    fitIntercept: whether to fit an intercept term. (default: True)
    labelCol: label column name. (default: label)
    lowerBoundsOnCoefficients: The lower bounds on coefficients if fitting under bound constrained optimization. The bound matrix must be compatible with the shape (1, number of features) for binomial regression, or (number of classes, number of features) for multinomial regression. (undefined)
    lowerBoundsOnIntercepts: The lower bounds on intercepts if fitting under bound constrained optimization. The bounds vector size must beequal with 1 for binomial regression, or the number oflasses for multinomial regression. (undefined)
    maxBlockSizeInMB: maximum memory in MB for stacking input data into blocks. Data is stacked within partitions. If more than remaining data size in a partition then it is adjusted to the data size. Default 0.0 represents choosing optimal value, depends on specific algorithm. Must be >= 0. (default: 0.0)
    maxIter: max number of iterations (>= 0). (default: 100, current: 10)
    predictionCol: prediction column name. (default: prediction)
    probabilityCol: Column name for predicted class conditional probabilities. Note: Not all models output well-calibrated probability estimates! These probabilities should be treated as confidences, not precise probabilities. (default: probability)
    rawPredictionCol: raw prediction (a.k.a. confidence) column name. (default: rawPrediction)
    regParam: regularization parameter (>= 0). (default: 0.0, current: 0.01)
    standardization: whether to standardize the training features before fitting the model. (default: True)
    threshold: Threshold in binary classification prediction, in range [0, 1]. If threshold and thresholds are both set, they must match.e.g. if threshold is p, then thresholds must be equal to [1-p, p]. (default: 0.5)
    thresholds: Thresholds in multi-class classification to adjust the probability of predicting each class. Array must have length equal to the number of classes, with values > 0, excepting that at most one value may be 0. The class with largest value p/t is predicted, where p is the original probability of that class and t is the class's threshold. (undefined)
    tol: the convergence tolerance for iterative algorithms (>= 0). (default: 1e-06)
    upperBoundsOnCoefficients: The upper bounds on coefficients if fitting under bound constrained optimization. The bound matrix must be compatible with the shape (1, number of features) for binomial regression, or (number of classes, number of features) for multinomial regression. (undefined)
    upperBoundsOnIntercepts: The upper bounds on intercepts if fitting under bound constrained optimization. The bound vector size must be equal with 1 for binomial regression, or the number of classes for multinomial regression. (undefined)
    weightCol: weight column name. If this is not set or empty, we treat all instance weights as 1.0. (undefined)
    
    features=[-1.0,1.5,1.3], label=1.0 -> prob=[0.0019392203169556147,0.9980607796830444], prediction=1.0
    features=[3.0,2.0,-0.1], label=0.0 -> prob=[0.995731919571047,0.004268080428952992], prediction=0.0
    features=[0.0,2.2,-1.5], label=1.0 -> prob=[0.01200463023637096,0.987995369763629], prediction=1.0


[Link to MLlib](https://spark.apache.org/docs/latest/ml-guide.html).

### Rapid Prototyping and Comparison of Basic Models with Lazy Predict

Lazy Predict enables rapid prototyping and comparison of multiple basic models without extensive manual coding or parameter tuning. 

This helps data scientists identify promising approaches and iterate on them more quickly.

```python
from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X = data.data
y= data.target

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.5,random_state =123)

clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

print(models)
```
```bash
| Model                          |   Accuracy |   Balanced Accuracy |   ROC AUC |   F1 Score |   Time Taken |
|:-------------------------------|-----------:|--------------------:|----------:|-----------:|-------------:|
| LinearSVC                      |   0.989474 |            0.987544 |  0.987544 |   0.989462 |    0.0150008 |
| SGDClassifier                  |   0.989474 |            0.987544 |  0.987544 |   0.989462 |    0.0109992 |
| MLPClassifier                  |   0.985965 |            0.986904 |  0.986904 |   0.985994 |    0.426     |
| Perceptron                     |   0.985965 |            0.984797 |  0.984797 |   0.985965 |    0.0120046 |
| LogisticRegression             |   0.985965 |            0.98269  |  0.98269  |   0.985934 |    0.0200036 |
| LogisticRegressionCV           |   0.985965 |            0.98269  |  0.98269  |   0.985934 |    0.262997  |
| SVC                            |   0.982456 |            0.979942 |  0.979942 |   0.982437 |    0.0140011 |
| CalibratedClassifierCV         |   0.982456 |            0.975728 |  0.975728 |   0.982357 |    0.0350015 |
| PassiveAggressiveClassifier    |   0.975439 |            0.974448 |  0.974448 |   0.975464 |    0.0130005 |
| LabelPropagation               |   0.975439 |            0.974448 |  0.974448 |   0.975464 |    0.0429988 |
| LabelSpreading                 |   0.975439 |            0.974448 |  0.974448 |   0.975464 |    0.0310006 |
| RandomForestClassifier         |   0.97193  |            0.969594 |  0.969594 |   0.97193  |    0.033     |
| GradientBoostingClassifier     |   0.97193  |            0.967486 |  0.967486 |   0.971869 |    0.166998  |
| QuadraticDiscriminantAnalysis  |   0.964912 |            0.966206 |  0.966206 |   0.965052 |    0.0119994 |
| HistGradientBoostingClassifier |   0.968421 |            0.964739 |  0.964739 |   0.968387 |    0.682003  |
| RidgeClassifierCV              |   0.97193  |            0.963272 |  0.963272 |   0.971736 |    0.0130029 |
| RidgeClassifier                |   0.968421 |            0.960525 |  0.960525 |   0.968242 |    0.0119977 |
| AdaBoostClassifier             |   0.961404 |            0.959245 |  0.959245 |   0.961444 |    0.204998  |
| ExtraTreesClassifier           |   0.961404 |            0.957138 |  0.957138 |   0.961362 |    0.0270066 |
| KNeighborsClassifier           |   0.961404 |            0.95503  |  0.95503  |   0.961276 |    0.0560005 |
| BaggingClassifier              |   0.947368 |            0.954577 |  0.954577 |   0.947882 |    0.0559971 |
| BernoulliNB                    |   0.950877 |            0.951003 |  0.951003 |   0.951072 |    0.0169988 |
| LinearDiscriminantAnalysis     |   0.961404 |            0.950816 |  0.950816 |   0.961089 |    0.0199995 |
| GaussianNB                     |   0.954386 |            0.949536 |  0.949536 |   0.954337 |    0.0139935 |
| NuSVC                          |   0.954386 |            0.943215 |  0.943215 |   0.954014 |    0.019989  |
| DecisionTreeClassifier         |   0.936842 |            0.933693 |  0.933693 |   0.936971 |    0.0170023 |
| NearestCentroid                |   0.947368 |            0.933506 |  0.933506 |   0.946801 |    0.0160074 |
| ExtraTreeClassifier            |   0.922807 |            0.912168 |  0.912168 |   0.922462 |    0.0109999 |
| CheckingClassifier             |   0.361404 |            0.5      |  0.5      |   0.191879 |    0.0170043 |
| DummyClassifier                |   0.512281 |            0.489598 |  0.489598 |   0.518924 |    0.0119965 |
```

[Link to Lazy Predict](https://github.com/shankarpandala/lazypredict).

### AutoGluon: Fast and Accurate ML in 3 Lines of Code

The traditional scikit-learn approach requires extensive manual work, including data preprocessing, model selection, and hyperparameter tuning.

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Preprocessing Pipeline
numeric_transformer = SimpleImputer(strategy='mean')
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])

# Machine Learning Pipeline
model = RandomForestClassifier()

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler()),
    ('model', model)
])

# Hyperparameter Tuning
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [5, 10, None],
    'model__min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
grid_search.predict(X_test)
```

In contrast, AutoGluon automates these tasks, allowing you to train and deploy accurate models in 3 lines of code.

```python
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(label="class").fit(train_data)
predictions = predictor.predict(test_data)
```

[Link to AutoGluon](https://bit.ly/45ljoOd).



Here is why using MLflow to log model is superior to using pickle to save model:

- Problem: Different models may require different versions of the same library, which can lead to conflicts. Manually tracking and setting up the correct environment for each model is time-consuming and error-prone.
- Solution: By automatically logging dependencies, MLflow ensures that anyone can recreate the exact environment needed to run the model.

- Problem: Often, the expected inputs and outputs of a model are not well-documented, making it difficult for others to use the model correctly.
- Solution: By defining a clear schema for inputs and outputs, MLflow ensures that anyone using the model knows exactly what data to provide and what to expect in return.

To illustrate the advantages of MLflow, let’s look at a simple implementation using a logistic regression model. Start with logging the model:


```python
import mlflow
from mlflow.models import infer_signature
import numpy as np
from sklearn.linear_model import LogisticRegression

with mlflow.start_run():
    X = np.array([-2, -1, 0, 1, 2, 1]).reshape(-1, 1)
    y = np.array([0, 0, 1, 1, 1, 0])
    lr = LogisticRegression()
    lr.fit(X, y)
    signature = infer_signature(X, lr.predict(X))

    model_info = mlflow.sklearn.log_model(
        sk_model=lr, artifact_path="model", signature=signature
    )

    print(f"Saving data to {model_info.model_uri}")
```

    Saving data to runs:/f8b0fc900aa14cf0ade8d0165c5a9f11/model


The output indicates where the model has been saved. To use the logged model later, you can use the `model_uri` as follows:


```python
import mlflow
import numpy as np

model_uri = "runs:/1e20d72afccf450faa3b8a9806a97e83/model"
sklearn_pyfunc = mlflow.pyfunc.load_model(model_uri=model_uri)

data = np.array([-4, 1, 0, 10, -2, 1]).reshape(-1, 1)

predictions = sklearn_pyfunc.predict(data)
```

Let's inspect the artifacts saved with the model:


```python
%cd mlruns/0/1e20d72afccf450faa3b8a9806a97e83/artifacts/model
%ls
```

    /Users/khuyentran/book/Efficient_Python_tricks_and_tools_for_data_scientists/Chapter5/mlruns/0/1e20d72afccf450faa3b8a9806a97e83/artifacts/model
    MLmodel           model.pkl         requirements.txt
    conda.yaml        python_env.yaml


The `MLmodel` file provides essential information about the model, including dependencies and input/output specifications:


```python
%cat MLmodel
```

    artifact_path: model
    flavors:
      python_function:
        env:
          conda: conda.yaml
          virtualenv: python_env.yaml
        loader_module: mlflow.sklearn
        model_path: model.pkl
        predict_fn: predict
        python_version: 3.11.6
      sklearn:
        code: null
        pickled_model: model.pkl
        serialization_format: cloudpickle
        sklearn_version: 1.4.1.post1
    mlflow_version: 2.15.0
    model_size_bytes: 722
    model_uuid: e7487bc3c4ab417c965144efcecaca2f
    run_id: 1e20d72afccf450faa3b8a9806a97e83
    signature:
      inputs: '[{"type": "tensor", "tensor-spec": {"dtype": "int64", "shape": [-1, 1]}}]'
      outputs: '[{"type": "tensor", "tensor-spec": {"dtype": "int64", "shape": [-1]}}]'
      params: null
    utc_time_created: '2024-08-02 20:58:16.516963'


The `conda.yaml` and `python_env.yaml` files outline the environment dependencies, ensuring that the model runs in a consistent setup:


```python
%cat conda.yaml
```

    channels:
    - conda-forge
    dependencies:
    - python=3.11.6
    - pip<=24.2
    - pip:
      - mlflow==2.15.0
      - cloudpickle==2.2.1
      - numpy==1.23.5
      - psutil==5.9.6
      - scikit-learn==1.4.1.post1
      - scipy==1.11.3
    name: mlflow-env



```python
%cat python_env.yaml

```

    python: 3.11.6
    build_dependencies:
    - pip==24.2
    - setuptools
    - wheel==0.40.0
    dependencies:
    - -r requirements.txt



```python
%cat requirements.txt
```

    mlflow==2.15.0
    cloudpickle==2.2.1
    numpy==1.23.5
    psutil==5.9.6
    scikit-learn==1.4.1.post1
    scipy==1.11.3
