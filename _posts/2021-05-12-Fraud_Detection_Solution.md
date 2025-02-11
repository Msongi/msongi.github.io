---
title: "Detecting Payment Card Fraud"
date: 2021-05-05
tags: [AWS SageMaker, data science, machine learning]
header:
  image: 
excerpt: "Data Wrangling, Data Science, Machine learning"
mathjax: "true"
---

In this section, we'll look at a credit card fraud detection dataset, and build a binary classification model that can identify transactions as either fraudulent or valid, based on provided, *historical* data. In a [2016 study](https://nilsonreport.com/upload/content_promo/The_Nilson_Report_10-17-2016.pdf), it was estimated that credit card fraud was responsible for over 20 billion dollars in loss, worldwide. Accurately detecting cases of fraud is an ongoing area of research.

<img src=notebook_ims/fraud_detection.png width=50% />

### Labeled Data

The payment fraud data set (Dal Pozzolo et al. 2015) was downloaded from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud/data). This has features and labels for thousands of credit card transactions, each of which is labeled as fraudulent or valid. In this notebook, we'd like to train a model based on the features of these transactions so that we can predict risky or fraudulent transactions in the future.

### Binary Classification

Since we have true labels to aim for, we'll take a **supervised learning** approach and train a binary classifier to sort data into one of our two transaction classes: fraudulent or valid.  We'll train a model on training data and see how well it generalizes on some test data.

The notebook will be broken down into a few steps:
* Loading and exploring the data
* Splitting the data into train/test sets
* Defining and training a LinearLearner, binary classifier
* Making improvements on the model
* Evaluating and comparing model test performance

### Making Improvements

A lot of this notebook will focus on making improvements, as discussed in [this SageMaker blog post](https://aws.amazon.com/blogs/machine-learning/train-faster-more-flexible-models-with-amazon-sagemaker-linear-learner/). Specifically, we'll address techniques for:

1. **Tuning a model's hyperparameters** and aiming for a specific metric, such as high recall or precision.
2. **Managing class imbalance**, which is when we have many more training examples in one class than another (in this case, many more valid transactions than fraudulent).

---

First, import the usual resources.


```python
import io
import os
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 

import boto3
import sagemaker
from sagemaker import get_execution_role

%matplotlib inline
```

I'm storing my **SageMaker variables** in the next cell:
* sagemaker_session: The SageMaker session we'll use for training models.
* bucket: The name of the default S3 bucket that we'll use for data storage.
* role: The IAM role that defines our data and model permissions.


```python
# sagemaker session, role
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# S3 bucket name
bucket = sagemaker_session.default_bucket()

```

## Loading and Exploring the Data

Next, I am loading the data and unzipping the data in the file `creditcardfraud.zip`. This directory will hold one csv file of all the transaction data, `creditcard.csv`.

As in previous notebooks, it's important to look at the distribution of data since this will inform how we develop a fraud detection model. We'll want to know: How many data points we have to work with, the number and type of features, and finally, the distribution of data over the classes (valid or fraudulent).


```python
# only have to run once
!wget https://s3.amazonaws.com/video.udacity-data.com/topher/2019/January/5c534768_creditcardfraud/creditcardfraud.zip
!unzip creditcardfraud
```


```python
# read in the csv file
local_data = 'creditcard.csv'

# print out some data
transaction_df = pd.read_csv(local_data)
print('Data shape (rows, cols): ', transaction_df.shape)
print()
transaction_df.head()
```

    Data shape (rows, cols):  (284807, 31)
    





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>-1.359807</td>
      <td>-0.072781</td>
      <td>2.536347</td>
      <td>1.378155</td>
      <td>-0.338321</td>
      <td>0.462388</td>
      <td>0.239599</td>
      <td>0.098698</td>
      <td>0.363787</td>
      <td>...</td>
      <td>-0.018307</td>
      <td>0.277838</td>
      <td>-0.110474</td>
      <td>0.066928</td>
      <td>0.128539</td>
      <td>-0.189115</td>
      <td>0.133558</td>
      <td>-0.021053</td>
      <td>149.62</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.191857</td>
      <td>0.266151</td>
      <td>0.166480</td>
      <td>0.448154</td>
      <td>0.060018</td>
      <td>-0.082361</td>
      <td>-0.078803</td>
      <td>0.085102</td>
      <td>-0.255425</td>
      <td>...</td>
      <td>-0.225775</td>
      <td>-0.638672</td>
      <td>0.101288</td>
      <td>-0.339846</td>
      <td>0.167170</td>
      <td>0.125895</td>
      <td>-0.008983</td>
      <td>0.014724</td>
      <td>2.69</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>-1.358354</td>
      <td>-1.340163</td>
      <td>1.773209</td>
      <td>0.379780</td>
      <td>-0.503198</td>
      <td>1.800499</td>
      <td>0.791461</td>
      <td>0.247676</td>
      <td>-1.514654</td>
      <td>...</td>
      <td>0.247998</td>
      <td>0.771679</td>
      <td>0.909412</td>
      <td>-0.689281</td>
      <td>-0.327642</td>
      <td>-0.139097</td>
      <td>-0.055353</td>
      <td>-0.059752</td>
      <td>378.66</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>-0.966272</td>
      <td>-0.185226</td>
      <td>1.792993</td>
      <td>-0.863291</td>
      <td>-0.010309</td>
      <td>1.247203</td>
      <td>0.237609</td>
      <td>0.377436</td>
      <td>-1.387024</td>
      <td>...</td>
      <td>-0.108300</td>
      <td>0.005274</td>
      <td>-0.190321</td>
      <td>-1.175575</td>
      <td>0.647376</td>
      <td>-0.221929</td>
      <td>0.062723</td>
      <td>0.061458</td>
      <td>123.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>-1.158233</td>
      <td>0.877737</td>
      <td>1.548718</td>
      <td>0.403034</td>
      <td>-0.407193</td>
      <td>0.095921</td>
      <td>0.592941</td>
      <td>-0.270533</td>
      <td>0.817739</td>
      <td>...</td>
      <td>-0.009431</td>
      <td>0.798278</td>
      <td>-0.137458</td>
      <td>0.141267</td>
      <td>-0.206010</td>
      <td>0.502292</td>
      <td>0.219422</td>
      <td>0.215153</td>
      <td>69.99</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>



### EXERCISE: Calculate the percentage of fraudulent data

Take a look at the distribution of this transaction data over the classes, valid and fraudulent. 

Complete the function `fraudulent_percentage`, below. Count up the number of data points in each class and calculate the *percentage* of the data points that are fraudulent.


```python
# Calculate the fraction of data points that are fraudulent
def fraudulent_percentage(transaction_df):
    '''Calculate the fraction of all data points that have a 'Class' label of 1; fraudulent.
       :param transaction_df: Dataframe of all transaction data points; has a column 'Class'
       :return: A fractional percentage of fraudulent data points/all points
    '''
    # counts for all classes
    counts = transaction_df['Class'].value_counts()
    
    # get fraudulent and valid cnts
    fraud_cnts = counts[1]
    valid_cnts = counts[0]
    
    # calculate percentage of fraudulent data
    fraud_percentage = fraud_cnts/(fraud_cnts+valid_cnts)
    
    return fraud_percentage

```

Test out your code by calling your function and printing the result.


```python
# call the function to calculate the fraud percentage
fraud_percentage = fraudulent_percentage(transaction_df)

print('Fraudulent percentage = ', fraud_percentage)
print('Total # of fraudulent pts: ', fraud_percentage*transaction_df.shape[0])
print('Out of (total) pts: ', transaction_df.shape[0])

```

    Fraudulent percentage =  0.001727485630620034
    Total # of fraudulent pts:  492.0
    Out of (total) pts:  284807


### EXERCISE: Split into train/test datasets

In this example, we'll want to evaluate the performance of a fraud classifier; training it on some training data and testing it on *test data* that it did not see during the training process. So, we'll need to split the data into separate training and test sets.

Complete the `train_test_split` function, below. This function should:
* Shuffle the transaction data, randomly
* Split it into two sets according to the parameter `train_frac`
* Get train/test features and labels
* Return the tuples: (train_features, train_labels), (test_features, test_labels)


```python
# split into train/test
def train_test_split(transaction_df, train_frac= 0.7, seed=1):
    '''Shuffle the data and randomly split into train and test sets;
       separate the class labels (the column in transaction_df) from the features.
       :param df: Dataframe of all credit card transaction data
       :param train_frac: The decimal fraction of data that should be training data
       :param seed: Random seed for shuffling and reproducibility, default = 1
       :return: Two tuples (in order): (train_features, train_labels), (test_features, test_labels)
       '''
    
    # convert the df into a matrix for ease of splitting
    df_matrix = transaction_df.as_matrix()
    
    # shuffle the data
    np.random.seed(seed)
    np.random.shuffle(df_matrix)
    
    # split the data
    train_size = int(df_matrix.shape[0] * train_frac)
    # features are all but last column
    train_features  = df_matrix[:train_size, :-1]
    # class labels *are* last column
    train_labels = df_matrix[:train_size, -1]
    # test data
    test_features = df_matrix[train_size:, :-1]
    test_labels = df_matrix[train_size:, -1]
    
    return (train_features, train_labels), (test_features, test_labels)

```

### Test Cell

In the cells below, I'm creating the train/test data and checking to see that result makes sense. The tests below test that the above function splits the data into the expected number of points and that the labels are indeed, class labels (0, 1).


```python
# get train/test data
(train_features, train_labels), (test_features, test_labels) = train_test_split(transaction_df, train_frac=0.7)

```


```python
# manual test

# for a split of 0.7:0.3 there should be ~2.33x as many training as test pts
print('Training data pts: ', len(train_features))
print('Test data pts: ', len(test_features))
print()

# take a look at first item and see that it aligns with first row of data
print('First item: \n', train_features[0])
print('Label: ', train_labels[0])
print()

# test split
assert len(train_features) > 2.333*len(test_features), \
        'Unexpected number of train/test points for a train_frac=0.7'
# test labels
assert np.all(train_labels)== 0 or np.all(train_labels)== 1, \
        'Train labels should be 0s or 1s.'
assert np.all(test_labels)== 0 or np.all(test_labels)== 1, \
        'Test labels should be 0s or 1s.'
print('Tests passed!')
```

    Training data pts:  199364
    Test data pts:  85443
    
    First item: 
     [ 1.19907000e+05 -6.11711999e-01 -7.69705324e-01 -1.49759145e-01
     -2.24876503e-01  2.02857736e+00 -2.01988711e+00  2.92491387e-01
     -5.23020325e-01  3.58468461e-01  7.00499612e-02 -8.54022784e-01
      5.47347360e-01  6.16448382e-01 -1.01785018e-01 -6.08491804e-01
     -2.88559430e-01 -6.06199260e-01 -9.00745518e-01 -2.01311157e-01
     -1.96039343e-01 -7.52077614e-02  4.55360454e-02  3.80739375e-01
      2.34403159e-02 -2.22068576e+00 -2.01145578e-01  6.65013699e-02
      2.21179560e-01  1.79000000e+00]
    Label:  0.0
    
    Tests passed!


---
# Modeling

Now that you've uploaded your training data, it's time to define and train a model!

In this notebook, you'll define and train the SageMaker, built-in algorithm, [LinearLearner](https://sagemaker.readthedocs.io/en/stable/linear_learner.html). 

A LinearLearner has two main applications:
1. For regression tasks in which a linear line is fit to some data points, and you want to produce a predicted output value given some data point (example: predicting house prices given square area).
2. For binary classification, in which a line is separating two classes of data and effectively outputs labels; either 1 for data that falls above the line or 0 for points that fall on or below the line.

<img src='notebook_ims/linear_separator.png' width=40% />

In this case, we'll be using it for case 2, and we'll train it to separate data into our two classes: valid or fraudulent. 

### EXERCISE: Create a LinearLearner Estimator

You've had some practice instantiating built-in models in SageMaker. All estimators require some constructor arguments to be passed in. See if you can complete this task, instantiating a LinearLearner estimator, using only the [LinearLearner documentation](https://sagemaker.readthedocs.io/en/stable/linear_learner.html) as a resource. This takes in a lot of arguments, but not all are required. My suggestion is to start with a simple model, utilizing default values where applicable. Later, we will discuss some specific hyperparameters and their use cases.

#### Instance Types

It is suggested that you use instances that are available in the free tier of usage: `'ml.c4.xlarge'` for training and `'ml.t2.medium'` for deployment.


```python
# import LinearLearner
from sagemaker import LinearLearner

# specify an output path
prefix = 'creditcard'
output_path = 's3://{}/{}'.format(bucket, prefix)

# instantiate LinearLearner
linear = LinearLearner(role=role,
                       train_instance_count=1, 
                       train_instance_type='ml.c4.xlarge',
                       predictor_type='binary_classifier',
                       output_path=output_path,
                       sagemaker_session=sagemaker_session,
                       epochs=15)

```

### EXERCISE: Convert data into a RecordSet format

Next, prepare the data for a built-in model by converting the train features and labels into numpy array's of float values. Then you can use the [record_set function](https://sagemaker.readthedocs.io/en/stable/linear_learner.html#sagemaker.LinearLearner.record_set) to format the data as a RecordSet and prepare it for training!


```python
# convert features/labels to numpy
train_x_np = train_features.astype('float32')
train_y_np = train_labels.astype('float32')

# create RecordSet
formatted_train_data = linear.record_set(train_x_np, labels=train_y_np)
```

### EXERCISE: Train the Estimator

After instantiating your estimator, train it with a call to `.fit()`, passing in the formatted training data.


```python
%%time 
# train the estimator on formatted training data
linear.fit(formatted_train_data)
```

    INFO:sagemaker:Creating training-job with name: linear-learner-2019-03-11-02-59-39-742


    2019-03-11 02:59:39 Starting - Starting the training job...
    2019-03-11 02:59:47 Starting - Launching requested ML instances......
    2019-03-11 03:01:00 Starting - Preparing the instances for training...
    2019-03-11 03:01:36 Downloading - Downloading input data...
    2019-03-11 03:01:44 Training - Downloading the training image..
    [31mDocker entrypoint called with argument(s): train[0m
    [31m[03/11/2019 03:02:22 INFO 140148523243328] Reading default configuration from /opt/amazon/lib/python2.7/site-packages/algorithm/default-input.json: {u'loss_insensitivity': u'0.01', u'epochs': u'15', u'init_bias': u'0.0', u'lr_scheduler_factor': u'auto', u'num_calibration_samples': u'10000000', u'accuracy_top_k': u'3', u'_num_kv_servers': u'auto', u'use_bias': u'true', u'num_point_for_scaler': u'10000', u'_log_level': u'info', u'quantile': u'0.5', u'bias_lr_mult': u'auto', u'lr_scheduler_step': u'auto', u'init_method': u'uniform', u'init_sigma': u'0.01', u'lr_scheduler_minimum_lr': u'auto', u'target_recall': u'0.8', u'num_models': u'auto', u'early_stopping_patience': u'3', u'momentum': u'auto', u'unbias_label': u'auto', u'wd': u'auto', u'optimizer': u'auto', u'_tuning_objective_metric': u'', u'early_stopping_tolerance': u'0.001', u'learning_rate': u'auto', u'_kvstore': u'auto', u'normalize_data': u'true', u'binary_classifier_model_selection_criteria': u'accuracy', u'use_lr_scheduler': u'true', u'target_precision': u'0.8', u'unbias_data': u'auto', u'init_scale': u'0.07', u'bias_wd_mult': u'auto', u'f_beta': u'1.0', u'mini_batch_size': u'1000', u'huber_delta': u'1.0', u'num_classes': u'1', u'beta_1': u'auto', u'loss': u'auto', u'beta_2': u'auto', u'_enable_profiler': u'false', u'normalize_label': u'auto', u'_num_gpus': u'auto', u'balance_multiclass_weights': u'false', u'positive_example_weight_mult': u'1.0', u'l1': u'auto', u'margin': u'1.0'}[0m
    [31m[03/11/2019 03:02:22 INFO 140148523243328] Reading provided configuration from /opt/ml/input/config/hyperparameters.json: {u'epochs': u'15', u'feature_dim': u'30', u'mini_batch_size': u'1000', u'predictor_type': u'binary_classifier'}[0m
    [31m[03/11/2019 03:02:22 INFO 140148523243328] Final configuration: {u'loss_insensitivity': u'0.01', u'epochs': u'15', u'feature_dim': u'30', u'init_bias': u'0.0', u'lr_scheduler_factor': u'auto', u'num_calibration_samples': u'10000000', u'accuracy_top_k': u'3', u'_num_kv_servers': u'auto', u'use_bias': u'true', u'num_point_for_scaler': u'10000', u'_log_level': u'info', u'quantile': u'0.5', u'bias_lr_mult': u'auto', u'lr_scheduler_step': u'auto', u'init_method': u'uniform', u'init_sigma': u'0.01', u'lr_scheduler_minimum_lr': u'auto', u'target_recall': u'0.8', u'num_models': u'auto', u'early_stopping_patience': u'3', u'momentum': u'auto', u'unbias_label': u'auto', u'wd': u'auto', u'optimizer': u'auto', u'_tuning_objective_metric': u'', u'early_stopping_tolerance': u'0.001', u'learning_rate': u'auto', u'_kvstore': u'auto', u'normalize_data': u'true', u'binary_classifier_model_selection_criteria': u'accuracy', u'use_lr_scheduler': u'true', u'target_precision': u'0.8', u'unbias_data': u'auto', u'init_scale': u'0.07', u'bias_wd_mult': u'auto', u'f_beta': u'1.0', u'mini_batch_size': u'1000', u'huber_delta': u'1.0', u'num_classes': u'1', u'predictor_type': u'binary_classifier', u'beta_1': u'auto', u'loss': u'auto', u'beta_2': u'auto', u'_enable_profiler': u'false', u'normalize_label': u'auto', u'_num_gpus': u'auto', u'balance_multiclass_weights': u'false', u'positive_example_weight_mult': u'1.0', u'l1': u'auto', u'margin': u'1.0'}[0m
    [31m[03/11/2019 03:02:22 WARNING 140148523243328] Loggers have already been setup.[0m
    [31mProcess 1 is a worker.[0m
    [31m[03/11/2019 03:02:22 INFO 140148523243328] Using default worker.[0m
    [31m[2019-03-11 03:02:22.445] [tensorio] [info] batch={"data_pipeline": "/opt/ml/input/data/train", "num_examples": 1000, "features": [{"name": "label_values", "shape": [1], "storage_type": "dense"}, {"name": "values", "shape": [30], "storage_type": "dense"}]}[0m
    [31m[2019-03-11 03:02:22.472] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 0, "duration": 28, "num_examples": 1}[0m
    [31m[03/11/2019 03:02:22 INFO 140148523243328] Create Store: local[0m
    [31m[2019-03-11 03:02:22.534] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 1, "duration": 60, "num_examples": 11}[0m
    [31m[03/11/2019 03:02:22 INFO 140148523243328] Scaler algorithm parameters
     <algorithm.scaler.ScalerAlgorithmStable object at 0x7f768ac7ba50>[0m
    [31m[03/11/2019 03:02:22 INFO 140148523243328] Scaling model computed with parameters:
     {'stdev_weight': [0m
    [31m[  4.75497891e+04   2.01225400e+00   1.72936726e+00   1.48752689e+00
       1.41830683e+00   1.42959750e+00   1.34760964e+00   1.27067423e+00
       1.24293745e+00   1.09265101e+00   1.05321789e+00   1.01260686e+00
       9.87991810e-01   1.00782645e+00   9.47202206e-01   9.02963459e-01
       8.68877888e-01   8.27179432e-01   8.36477458e-01   8.07050884e-01
       8.00110519e-01   7.55493522e-01   7.21427202e-01   6.25614405e-01
       6.10876381e-01   5.16283095e-01   4.88118291e-01   4.35698181e-01
       3.69419903e-01   2.47155548e+02][0m
    [31m<NDArray 30 @cpu(0)>, 'stdev_label': None, 'mean_label': None, 'mean_weight': [0m
    [31m[  9.44802812e+04  -1.04726264e-02  -1.43008800e-02   1.28451567e-02
       1.87512934e-02  -2.48281248e-02   5.86199807e-03  -7.13069551e-03
      -7.39883492e-03   1.20382467e-02   6.10911567e-03  -3.16866231e-03
       8.64854374e-04   2.46435311e-03   1.56665407e-02   1.12619074e-02
      -4.91584092e-03  -1.56447978e-03   2.45723873e-03   2.82235094e-04
      -3.25949211e-03   6.57527940e-03   3.11945518e-03   6.22356636e-03
      -6.13171898e-04  -3.88089707e-03   1.16021503e-02  -3.21021304e-03
      -5.27510792e-03   8.94287567e+01][0m
    [31m<NDArray 30 @cpu(0)>}[0m
    [31m[03/11/2019 03:02:22 INFO 140148523243328] nvidia-smi took: 0.0251679420471 secs to identify 0 gpus[0m
    [31m[03/11/2019 03:02:22 INFO 140148523243328] Number of GPUs being used: 0[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 11, "sum": 11.0, "min": 11}, "Number of Batches Since Last Reset": {"count": 1, "max": 0, "sum": 0.0, "min": 0}, "Number of Records Since Last Reset": {"count": 1, "max": 0, "sum": 0.0, "min": 0}, "Total Batches Seen": {"count": 1, "max": 12, "sum": 12.0, "min": 12}, "Total Records Seen": {"count": 1, "max": 12000, "sum": 12000.0, "min": 12000}, "Max Records Seen Between Resets": {"count": 1, "max": 11000, "sum": 11000.0, "min": 11000}, "Reset Count": {"count": 1, "max": 2, "sum": 2.0, "min": 2}}, "EndTime": 1552273342.642219, "Dimensions": {"Host": "algo-1", "Meta": "init_train_data_iter", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1552273342.642184}
    [0m
    
    2019-03-11 03:02:19 Training - Training image download completed. Training in progress.[31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.12043031547776419, "sum": 0.12043031547776419, "min": 0.12043031547776419}}, "EndTime": 1552273348.875526, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552273348.875466}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.11774842056197737, "sum": 0.11774842056197737, "min": 0.11774842056197737}}, "EndTime": 1552273348.875605, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552273348.875592}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.12036449890520105, "sum": 0.12036449890520105, "min": 0.12036449890520105}}, "EndTime": 1552273348.875657, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552273348.875644}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.11797112150527723, "sum": 0.11797112150527723, "min": 0.11797112150527723}}, "EndTime": 1552273348.875705, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552273348.875695}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012310507309758783, "sum": 0.012310507309758783, "min": 0.012310507309758783}}, "EndTime": 1552273348.875735, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552273348.875728}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012591959382159327, "sum": 0.012591959382159327, "min": 0.012591959382159327}}, "EndTime": 1552273348.875774, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552273348.875761}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012322717984342695, "sum": 0.012322717984342695, "min": 0.012322717984342695}}, "EndTime": 1552273348.875828, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552273348.875813}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012404795770743984, "sum": 0.012404795770743984, "min": 0.012404795770743984}}, "EndTime": 1552273348.87588, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552273348.875866}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.12042285891393921, "sum": 0.12042285891393921, "min": 0.12042285891393921}}, "EndTime": 1552273348.875911, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552273348.875903}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1179497050855627, "sum": 0.1179497050855627, "min": 0.1179497050855627}}, "EndTime": 1552273348.875937, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552273348.87593}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.12053752546454195, "sum": 0.12053752546454195, "min": 0.12053752546454195}}, "EndTime": 1552273348.875966, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552273348.875959}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.11788512564424294, "sum": 0.11788512564424294, "min": 0.11788512564424294}}, "EndTime": 1552273348.875993, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552273348.875986}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012449912267788571, "sum": 0.012449912267788571, "min": 0.012449912267788571}}, "EndTime": 1552273348.876029, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552273348.876016}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01263913904151051, "sum": 0.01263913904151051, "min": 0.01263913904151051}}, "EndTime": 1552273348.876086, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552273348.876071}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012591561175283775, "sum": 0.012591561175283775, "min": 0.012591561175283775}}, "EndTime": 1552273348.876138, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552273348.876124}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012598671647153758, "sum": 0.012598671647153758, "min": 0.012598671647153758}}, "EndTime": 1552273348.876188, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552273348.876177}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1229229508883989, "sum": 0.1229229508883989, "min": 0.1229229508883989}}, "EndTime": 1552273348.87623, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552273348.876223}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1202098897617666, "sum": 0.1202098897617666, "min": 0.1202098897617666}}, "EndTime": 1552273348.876256, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552273348.876249}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.12281814978709772, "sum": 0.12281814978709772, "min": 0.12281814978709772}}, "EndTime": 1552273348.876281, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552273348.876274}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1203641751639208, "sum": 0.1203641751639208, "min": 0.1203641751639208}}, "EndTime": 1552273348.876335, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552273348.876322}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.019713981734822743, "sum": 0.019713981734822743, "min": 0.019713981734822743}}, "EndTime": 1552273348.876387, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552273348.876372}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.019672431648339157, "sum": 0.019672431648339157, "min": 0.019672431648339157}}, "EndTime": 1552273348.876437, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552273348.876422}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.019612770365298394, "sum": 0.019612770365298394, "min": 0.019612770365298394}}, "EndTime": 1552273348.876493, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552273348.876477}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.019638996485002974, "sum": 0.019638996485002974, "min": 0.019638996485002974}}, "EndTime": 1552273348.876545, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552273348.87653}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1237417889312284, "sum": 0.1237417889312284, "min": 0.1237417889312284}}, "EndTime": 1552273348.876597, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552273348.876582}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.12108165967164926, "sum": 0.12108165967164926, "min": 0.12108165967164926}}, "EndTime": 1552273348.87665, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552273348.876635}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.12376727658180735, "sum": 0.12376727658180735, "min": 0.12376727658180735}}, "EndTime": 1552273348.876713, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552273348.876698}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.12122506574889523, "sum": 0.12122506574889523, "min": 0.12122506574889523}}, "EndTime": 1552273348.876769, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552273348.876753}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.020484194981991947, "sum": 0.020484194981991947, "min": 0.020484194981991947}}, "EndTime": 1552273348.876824, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552273348.876809}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.020433710982302327, "sum": 0.020433710982302327, "min": 0.020433710982302327}}, "EndTime": 1552273348.876879, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552273348.876864}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.020589645515434706, "sum": 0.020589645515434706, "min": 0.020589645515434706}}, "EndTime": 1552273348.876932, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552273348.876917}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.020449347862346687, "sum": 0.020449347862346687, "min": 0.020449347862346687}}, "EndTime": 1552273348.876984, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552273348.87697}
    [0m
    [31m[03/11/2019 03:02:28 INFO 140148523243328] #quality_metric: host=algo-1, epoch=0, train binary_classification_cross_entropy_objective <loss>=0.120430315478[0m
    [31m[03/11/2019 03:02:28 INFO 140148523243328] #early_stopping_criteria_metric: host=algo-1, epoch=0, criteria=binary_classification_cross_entropy_objective, value=0.0123105073098[0m
    [31m[03/11/2019 03:02:28 INFO 140148523243328] Epoch 0: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:02:28 INFO 140148523243328] #progress_metric: host=algo-1, completed 6 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 212, "sum": 212.0, "min": 212}, "Total Records Seen": {"count": 1, "max": 211364, "sum": 211364.0, "min": 211364}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 3, "sum": 3.0, "min": 3}}, "EndTime": 1552273348.880002, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552273342.642435}
    [0m
    [31m[03/11/2019 03:02:28 INFO 140148523243328] #throughput_metric: host=algo-1, train throughput=31961.3338364 records/second[0m
    [31m[2019-03-11 03:02:28.880] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 2, "duration": 6237, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01697396398668912, "sum": 0.01697396398668912, "min": 0.01697396398668912}}, "EndTime": 1552273354.713502, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552273354.713423}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.014988560247660881, "sum": 0.014988560247660881, "min": 0.014988560247660881}}, "EndTime": 1552273354.713591, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552273354.713576}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.016968431084599327, "sum": 0.016968431084599327, "min": 0.016968431084599327}}, "EndTime": 1552273354.713644, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552273354.713631}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.014997809695239043, "sum": 0.014997809695239043, "min": 0.014997809695239043}}, "EndTime": 1552273354.713689, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552273354.713678}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005413321109123566, "sum": 0.005413321109123566, "min": 0.005413321109123566}}, "EndTime": 1552273354.713748, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552273354.713718}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.006144253869377189, "sum": 0.006144253869377189, "min": 0.006144253869377189}}, "EndTime": 1552273354.713794, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552273354.713781}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005583528396097859, "sum": 0.005583528396097859, "min": 0.005583528396097859}}, "EndTime": 1552273354.713839, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552273354.713827}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005664390350780895, "sum": 0.005664390350780895, "min": 0.005664390350780895}}, "EndTime": 1552273354.713878, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552273354.713868}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.017126370190376012, "sum": 0.017126370190376012, "min": 0.017126370190376012}}, "EndTime": 1552273354.713916, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552273354.713907}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.015161259157573758, "sum": 0.015161259157573758, "min": 0.015161259157573758}}, "EndTime": 1552273354.713954, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552273354.713944}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.017126536081783737, "sum": 0.017126536081783737, "min": 0.017126536081783737}}, "EndTime": 1552273354.713991, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552273354.713981}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.015162341254440385, "sum": 0.015162341254440385, "min": 0.015162341254440385}}, "EndTime": 1552273354.714029, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552273354.714019}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005557677248015476, "sum": 0.005557677248015476, "min": 0.005557677248015476}}, "EndTime": 1552273354.714066, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552273354.714056}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005842554873258026, "sum": 0.005842554873258026, "min": 0.005842554873258026}}, "EndTime": 1552273354.714103, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552273354.714093}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005539924988195524, "sum": 0.005539924988195524, "min": 0.005539924988195524}}, "EndTime": 1552273354.714139, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552273354.71413}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005890578771356362, "sum": 0.005890578771356362, "min": 0.005890578771356362}}, "EndTime": 1552273354.714176, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552273354.714167}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.021226986760470138, "sum": 0.021226986760470138, "min": 0.021226986760470138}}, "EndTime": 1552273354.714213, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552273354.714203}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.019429727343458627, "sum": 0.019429727343458627, "min": 0.019429727343458627}}, "EndTime": 1552273354.71425, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552273354.71424}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.02121443661253656, "sum": 0.02121443661253656, "min": 0.02121443661253656}}, "EndTime": 1552273354.714286, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552273354.714277}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0194394117863334, "sum": 0.0194394117863334, "min": 0.0194394117863334}}, "EndTime": 1552273354.714332, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552273354.714321}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011853274800669607, "sum": 0.011853274800669607, "min": 0.011853274800669607}}, "EndTime": 1552273354.714373, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552273354.714363}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011885794957678521, "sum": 0.011885794957678521, "min": 0.011885794957678521}}, "EndTime": 1552273354.714412, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552273354.714402}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011870770789870066, "sum": 0.011870770789870066, "min": 0.011870770789870066}}, "EndTime": 1552273354.71445, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552273354.71444}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011879496791254935, "sum": 0.011879496791254935, "min": 0.011879496791254935}}, "EndTime": 1552273354.714488, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552273354.714478}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.02215121567309202, "sum": 0.02215121567309202, "min": 0.02215121567309202}}, "EndTime": 1552273354.714527, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552273354.714517}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.020372232250232793, "sum": 0.020372232250232793, "min": 0.020372232250232793}}, "EndTime": 1552273354.714564, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552273354.714555}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.02215141379294084, "sum": 0.02215141379294084, "min": 0.02215141379294084}}, "EndTime": 1552273354.714603, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552273354.714593}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.02038332456080758, "sum": 0.02038332456080758, "min": 0.02038332456080758}}, "EndTime": 1552273354.71464, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552273354.71463}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012853264628343247, "sum": 0.012853264628343247, "min": 0.012853264628343247}}, "EndTime": 1552273354.714676, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552273354.714666}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012862902701200551, "sum": 0.012862902701200551, "min": 0.012862902701200551}}, "EndTime": 1552273354.714713, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552273354.714703}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012854538045935894, "sum": 0.012854538045935894, "min": 0.012854538045935894}}, "EndTime": 1552273354.714749, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552273354.71474}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012862664958939481, "sum": 0.012862664958939481, "min": 0.012862664958939481}}, "EndTime": 1552273354.714786, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552273354.714777}
    [0m
    [31m[03/11/2019 03:02:34 INFO 140148523243328] #quality_metric: host=algo-1, epoch=1, train binary_classification_cross_entropy_objective <loss>=0.0169739639867[0m
    [31m[03/11/2019 03:02:34 INFO 140148523243328] #early_stopping_criteria_metric: host=algo-1, epoch=1, criteria=binary_classification_cross_entropy_objective, value=0.00541332110912[0m
    [31m[03/11/2019 03:02:34 INFO 140148523243328] Epoch 1: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:02:34 INFO 140148523243328] #progress_metric: host=algo-1, completed 13 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 412, "sum": 412.0, "min": 412}, "Total Records Seen": {"count": 1, "max": 410728, "sum": 410728.0, "min": 410728}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 4, "sum": 4.0, "min": 4}}, "EndTime": 1552273354.71732, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552273348.880271}
    [0m
    [31m[03/11/2019 03:02:34 INFO 140148523243328] #throughput_metric: host=algo-1, train throughput=34154.1403574 records/second[0m
    [31m[2019-03-11 03:02:34.717] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 3, "duration": 5837, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.010330928617985404, "sum": 0.010330928617985404, "min": 0.010330928617985404}}, "EndTime": 1552273360.739341, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552273360.73925}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.008765228896883864, "sum": 0.008765228896883864, "min": 0.008765228896883864}}, "EndTime": 1552273360.739459, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552273360.73943}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.010327125379188576, "sum": 0.010327125379188576, "min": 0.010327125379188576}}, "EndTime": 1552273360.739516, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552273360.739502}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.008771104021887084, "sum": 0.008771104021887084, "min": 0.008771104021887084}}, "EndTime": 1552273360.739565, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552273360.739552}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005136206356724303, "sum": 0.005136206356724303, "min": 0.005136206356724303}}, "EndTime": 1552273360.73961, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552273360.739599}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005824846665388975, "sum": 0.005824846665388975, "min": 0.005824846665388975}}, "EndTime": 1552273360.739652, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552273360.739641}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005184625139787569, "sum": 0.005184625139787569, "min": 0.005184625139787569}}, "EndTime": 1552273360.739694, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552273360.739683}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005772609489721869, "sum": 0.005772609489721869, "min": 0.005772609489721869}}, "EndTime": 1552273360.739735, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552273360.739724}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.010579428452343198, "sum": 0.010579428452343198, "min": 0.010579428452343198}}, "EndTime": 1552273360.739777, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552273360.739766}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.009065929326579798, "sum": 0.009065929326579798, "min": 0.009065929326579798}}, "EndTime": 1552273360.739818, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552273360.739807}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.010578739535269426, "sum": 0.010578739535269426, "min": 0.010578739535269426}}, "EndTime": 1552273360.739859, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552273360.739849}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.009066744816363157, "sum": 0.009066744816363157, "min": 0.009066744816363157}}, "EndTime": 1552273360.739901, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552273360.739891}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005415446985606572, "sum": 0.005415446985606572, "min": 0.005415446985606572}}, "EndTime": 1552273360.739942, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552273360.739932}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005902450287761401, "sum": 0.005902450287761401, "min": 0.005902450287761401}}, "EndTime": 1552273360.739983, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552273360.739972}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005396837090727073, "sum": 0.005396837090727073, "min": 0.005396837090727073}}, "EndTime": 1552273360.740027, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552273360.740017}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005898476717460095, "sum": 0.005898476717460095, "min": 0.005898476717460095}}, "EndTime": 1552273360.740069, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552273360.740058}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01543734490332292, "sum": 0.01543734490332292, "min": 0.01543734490332292}}, "EndTime": 1552273360.740109, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552273360.740099}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.014191929726145375, "sum": 0.014191929726145375, "min": 0.014191929726145375}}, "EndTime": 1552273360.740149, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552273360.740139}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.015432025216931674, "sum": 0.015432025216931674, "min": 0.015432025216931674}}, "EndTime": 1552273360.74019, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552273360.740179}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.014196535096096634, "sum": 0.014196535096096634, "min": 0.014196535096096634}}, "EndTime": 1552273360.74023, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552273360.74022}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011893838788995791, "sum": 0.011893838788995791, "min": 0.011893838788995791}}, "EndTime": 1552273360.740271, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552273360.74026}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011907273306918504, "sum": 0.011907273306918504, "min": 0.011907273306918504}}, "EndTime": 1552273360.740311, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552273360.7403}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011899466008397202, "sum": 0.011899466008397202, "min": 0.011899466008397202}}, "EndTime": 1552273360.740351, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552273360.74034}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011901343090450345, "sum": 0.011901343090450345, "min": 0.011901343090450345}}, "EndTime": 1552273360.740393, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552273360.740382}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.016371653041647907, "sum": 0.016371653041647907, "min": 0.016371653041647907}}, "EndTime": 1552273360.740435, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552273360.740424}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.015148403155743777, "sum": 0.015148403155743777, "min": 0.015148403155743777}}, "EndTime": 1552273360.740476, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552273360.740465}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.016371930594420315, "sum": 0.016371930594420315, "min": 0.016371930594420315}}, "EndTime": 1552273360.740517, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552273360.740507}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01515225470605208, "sum": 0.01515225470605208, "min": 0.01515225470605208}}, "EndTime": 1552273360.740557, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552273360.740547}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012857517996625085, "sum": 0.012857517996625085, "min": 0.012857517996625085}}, "EndTime": 1552273360.740598, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552273360.740587}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012874755165085721, "sum": 0.012874755165085721, "min": 0.012874755165085721}}, "EndTime": 1552273360.740639, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552273360.740628}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012858834195975681, "sum": 0.012858834195975681, "min": 0.012858834195975681}}, "EndTime": 1552273360.74068, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552273360.740669}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012874105859042412, "sum": 0.012874105859042412, "min": 0.012874105859042412}}, "EndTime": 1552273360.740719, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552273360.740709}
    [0m
    [31m[03/11/2019 03:02:40 INFO 140148523243328] #quality_metric: host=algo-1, epoch=2, train binary_classification_cross_entropy_objective <loss>=0.010330928618[0m
    [31m[03/11/2019 03:02:40 INFO 140148523243328] #early_stopping_criteria_metric: host=algo-1, epoch=2, criteria=binary_classification_cross_entropy_objective, value=0.00513620635672[0m
    [31m[03/11/2019 03:02:40 INFO 140148523243328] Epoch 2: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:02:40 INFO 140148523243328] #progress_metric: host=algo-1, completed 20 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 612, "sum": 612.0, "min": 612}, "Total Records Seen": {"count": 1, "max": 610092, "sum": 610092.0, "min": 610092}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 5, "sum": 5.0, "min": 5}}, "EndTime": 1552273360.744301, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552273354.717618}
    [0m
    [31m[03/11/2019 03:02:40 INFO 140148523243328] #throughput_metric: host=algo-1, train throughput=33079.5443157 records/second[0m
    [31m[2019-03-11 03:02:40.744] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 4, "duration": 6026, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.008034130657138537, "sum": 0.008034130657138537, "min": 0.008034130657138537}}, "EndTime": 1552273366.726401, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552273366.726317}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.006699231958868516, "sum": 0.006699231958868516, "min": 0.006699231958868516}}, "EndTime": 1552273366.726492, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552273366.726476}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.008031785320396998, "sum": 0.008031785320396998, "min": 0.008031785320396998}}, "EndTime": 1552273366.72667, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552273366.726652}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.006702904769523659, "sum": 0.006702904769523659, "min": 0.006702904769523659}}, "EndTime": 1552273366.726718, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552273366.726706}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0050371206441716335, "sum": 0.0050371206441716335, "min": 0.0050371206441716335}}, "EndTime": 1552273366.726762, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552273366.726751}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.006105847063301197, "sum": 0.006105847063301197, "min": 0.006105847063301197}}, "EndTime": 1552273366.726802, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552273366.726792}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005057589868504797, "sum": 0.005057589868504797, "min": 0.005057589868504797}}, "EndTime": 1552273366.726842, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552273366.726831}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.006192843649405331, "sum": 0.006192843649405331, "min": 0.006192843649405331}}, "EndTime": 1552273366.72688, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552273366.72687}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.008379334282036403, "sum": 0.008379334282036403, "min": 0.008379334282036403}}, "EndTime": 1552273366.72692, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552273366.726909}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.007128520195208602, "sum": 0.007128520195208602, "min": 0.007128520195208602}}, "EndTime": 1552273366.726959, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552273366.726949}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.008378759724410934, "sum": 0.008378759724410934, "min": 0.008378759724410934}}, "EndTime": 1552273366.726996, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552273366.726987}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0071292423375287845, "sum": 0.0071292423375287845, "min": 0.0071292423375287845}}, "EndTime": 1552273366.727044, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552273366.727034}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005350434124469757, "sum": 0.005350434124469757, "min": 0.005350434124469757}}, "EndTime": 1552273366.72709, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552273366.727079}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005897634128230301, "sum": 0.005897634128230301, "min": 0.005897634128230301}}, "EndTime": 1552273366.72713, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552273366.72712}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005347798913267989, "sum": 0.005347798913267989, "min": 0.005347798913267989}}, "EndTime": 1552273366.727167, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552273366.727158}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.00589653688789013, "sum": 0.00589653688789013, "min": 0.00589653688789013}}, "EndTime": 1552273366.72721, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552273366.7272}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.013688772067352755, "sum": 0.013688772067352755, "min": 0.013688772067352755}}, "EndTime": 1552273366.727254, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552273366.727244}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012770949257079082, "sum": 0.012770949257079082, "min": 0.012770949257079082}}, "EndTime": 1552273366.727298, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552273366.727287}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.013685778299168725, "sum": 0.013685778299168725, "min": 0.013685778299168725}}, "EndTime": 1552273366.727341, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552273366.727331}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012773823243289737, "sum": 0.012773823243289737, "min": 0.012773823243289737}}, "EndTime": 1552273366.727387, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552273366.727376}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011883480945424218, "sum": 0.011883480945424218, "min": 0.011883480945424218}}, "EndTime": 1552273366.727474, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552273366.727458}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011918503241323347, "sum": 0.011918503241323347, "min": 0.011918503241323347}}, "EndTime": 1552273366.727524, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552273366.727513}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011878312025237921, "sum": 0.011878312025237921, "min": 0.011878312025237921}}, "EndTime": 1552273366.727571, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552273366.727558}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011914209698911887, "sum": 0.011914209698911887, "min": 0.011914209698911887}}, "EndTime": 1552273366.727605, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552273366.727597}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.014625513352341388, "sum": 0.014625513352341388, "min": 0.014625513352341388}}, "EndTime": 1552273366.72763, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552273366.727623}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.013732083848972416, "sum": 0.013732083848972416, "min": 0.013732083848972416}}, "EndTime": 1552273366.727655, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552273366.727649}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01462572515430163, "sum": 0.01462572515430163, "min": 0.01462572515430163}}, "EndTime": 1552273366.727689, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552273366.72768}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.013733733051386312, "sum": 0.013733733051386312, "min": 0.013733733051386312}}, "EndTime": 1552273366.72772, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552273366.727713}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012857922017873832, "sum": 0.012857922017873832, "min": 0.012857922017873832}}, "EndTime": 1552273366.727764, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552273366.72775}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012884794590461194, "sum": 0.012884794590461194, "min": 0.012884794590461194}}, "EndTime": 1552273366.727816, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552273366.727806}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012859059595582473, "sum": 0.012859059595582473, "min": 0.012859059595582473}}, "EndTime": 1552273366.727851, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552273366.727843}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012884603715422166, "sum": 0.012884603715422166, "min": 0.012884603715422166}}, "EndTime": 1552273366.727882, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552273366.727875}
    [0m
    [31m[03/11/2019 03:02:46 INFO 140148523243328] #quality_metric: host=algo-1, epoch=3, train binary_classification_cross_entropy_objective <loss>=0.00803413065714[0m
    [31m[03/11/2019 03:02:46 INFO 140148523243328] #early_stopping_criteria_metric: host=algo-1, epoch=3, criteria=binary_classification_cross_entropy_objective, value=0.00503712064417[0m
    [31m[03/11/2019 03:02:46 INFO 140148523243328] Epoch 3: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:02:46 INFO 140148523243328] #progress_metric: host=algo-1, completed 26 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 812, "sum": 812.0, "min": 812}, "Total Records Seen": {"count": 1, "max": 809456, "sum": 809456.0, "min": 809456}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 6, "sum": 6.0, "min": 6}}, "EndTime": 1552273366.731603, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552273360.744559}
    [0m
    [31m[03/11/2019 03:02:46 INFO 140148523243328] #throughput_metric: host=algo-1, train throughput=33298.493052 records/second[0m
    [31m[2019-03-11 03:02:46.731] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 5, "duration": 5986, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0069169620389315355, "sum": 0.0069169620389315355, "min": 0.0069169620389315355}}, "EndTime": 1552273372.792329, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552273372.792269}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.00574524320429893, "sum": 0.00574524320429893, "min": 0.00574524320429893}}, "EndTime": 1552273372.7924, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552273372.792387}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.00691596810062926, "sum": 0.00691596810062926, "min": 0.00691596810062926}}, "EndTime": 1552273372.792453, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552273372.792437}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005747290889222418, "sum": 0.005747290889222418, "min": 0.005747290889222418}}, "EndTime": 1552273372.792556, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552273372.792502}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0049067022006715364, "sum": 0.0049067022006715364, "min": 0.0049067022006715364}}, "EndTime": 1552273372.792628, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552273372.792609}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005987407533891836, "sum": 0.005987407533891836, "min": 0.005987407533891836}}, "EndTime": 1552273372.792685, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552273372.79267}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004837448609234699, "sum": 0.004837448609234699, "min": 0.004837448609234699}}, "EndTime": 1552273372.792737, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552273372.792723}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005820065221295285, "sum": 0.005820065221295285, "min": 0.005820065221295285}}, "EndTime": 1552273372.792792, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552273372.792777}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.007338020905777437, "sum": 0.007338020905777437, "min": 0.007338020905777437}}, "EndTime": 1552273372.792849, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552273372.792833}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.006271388709245615, "sum": 0.006271388709245615, "min": 0.006271388709245615}}, "EndTime": 1552273372.792904, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552273372.792889}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.007337606277897131, "sum": 0.007337606277897131, "min": 0.007337606277897131}}, "EndTime": 1552273372.79297, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552273372.792953}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0062720328922846805, "sum": 0.0062720328922846805, "min": 0.0062720328922846805}}, "EndTime": 1552273372.793036, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552273372.793019}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005339479718076524, "sum": 0.005339479718076524, "min": 0.005339479718076524}}, "EndTime": 1552273372.793107, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552273372.793091}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005877964826984022, "sum": 0.005877964826984022, "min": 0.005877964826984022}}, "EndTime": 1552273372.793172, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552273372.793156}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0053357686253648305, "sum": 0.0053357686253648305, "min": 0.0053357686253648305}}, "EndTime": 1552273372.793229, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552273372.793213}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005874441825864303, "sum": 0.005874441825864303, "min": 0.005874441825864303}}, "EndTime": 1552273372.793282, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552273372.793267}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01294359584309947, "sum": 0.01294359584309947, "min": 0.01294359584309947}}, "EndTime": 1552273372.793336, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552273372.793321}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012251624374533419, "sum": 0.012251624374533419, "min": 0.012251624374533419}}, "EndTime": 1552273372.793391, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552273372.793376}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012941720984089914, "sum": 0.012941720984089914, "min": 0.012941720984089914}}, "EndTime": 1552273372.793446, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552273372.793431}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012253663547074975, "sum": 0.012253663547074975, "min": 0.012253663547074975}}, "EndTime": 1552273372.793502, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552273372.793487}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011882697144345422, "sum": 0.011882697144345422, "min": 0.011882697144345422}}, "EndTime": 1552273372.793568, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552273372.79355}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01192718984074329, "sum": 0.01192718984074329, "min": 0.01192718984074329}}, "EndTime": 1552273372.793634, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552273372.793617}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011877108363649953, "sum": 0.011877108363649953, "min": 0.011877108363649953}}, "EndTime": 1552273372.793694, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552273372.793677}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011924551403702204, "sum": 0.011924551403702204, "min": 0.011924551403702204}}, "EndTime": 1552273372.793783, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552273372.793765}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.013881827797721978, "sum": 0.013881827797721978, "min": 0.013881827797721978}}, "EndTime": 1552273372.793845, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552273372.793828}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01321468453311441, "sum": 0.01321468453311441, "min": 0.01321468453311441}}, "EndTime": 1552273372.793908, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552273372.793892}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.013881908687514875, "sum": 0.013881908687514875, "min": 0.013881908687514875}}, "EndTime": 1552273372.793964, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552273372.793949}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.013215637271727749, "sum": 0.013215637271727749, "min": 0.013215637271727749}}, "EndTime": 1552273372.79402, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552273372.794005}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01285734530729265, "sum": 0.01285734530729265, "min": 0.01285734530729265}}, "EndTime": 1552273372.794076, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552273372.794061}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012893029334557117, "sum": 0.012893029334557117, "min": 0.012893029334557117}}, "EndTime": 1552273372.79413, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552273372.794115}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012858692809564984, "sum": 0.012858692809564984, "min": 0.012858692809564984}}, "EndTime": 1552273372.79419, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552273372.794174}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012895550845855445, "sum": 0.012895550845855445, "min": 0.012895550845855445}}, "EndTime": 1552273372.79425, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552273372.794235}
    [0m
    [31m[03/11/2019 03:02:52 INFO 140148523243328] #quality_metric: host=algo-1, epoch=4, train binary_classification_cross_entropy_objective <loss>=0.00691696203893[0m
    [31m[03/11/2019 03:02:52 INFO 140148523243328] #early_stopping_criteria_metric: host=algo-1, epoch=4, criteria=binary_classification_cross_entropy_objective, value=0.00483744860923[0m
    [31m[03/11/2019 03:02:52 INFO 140148523243328] Epoch 4: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:02:52 INFO 140148523243328] #progress_metric: host=algo-1, completed 33 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 1012, "sum": 1012.0, "min": 1012}, "Total Records Seen": {"count": 1, "max": 1008820, "sum": 1008820.0, "min": 1008820}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 7, "sum": 7.0, "min": 7}}, "EndTime": 1552273372.796735, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552273366.731894}
    [0m
    [31m[03/11/2019 03:02:52 INFO 140148523243328] #throughput_metric: host=algo-1, train throughput=32871.4942419 records/second[0m
    [31m[2019-03-11 03:02:52.796] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 6, "duration": 6064, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.006276343655945668, "sum": 0.006276343655945668, "min": 0.006276343655945668}}, "EndTime": 1552273378.554432, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552273378.554368}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005225318925464573, "sum": 0.005225318925464573, "min": 0.005225318925464573}}, "EndTime": 1552273378.554516, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552273378.554503}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.006276083052457877, "sum": 0.006276083052457877, "min": 0.006276083052457877}}, "EndTime": 1552273378.554551, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552273378.554543}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0052264984804182195, "sum": 0.0052264984804182195, "min": 0.0052264984804182195}}, "EndTime": 1552273378.554583, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552273378.554575}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004745869061305894, "sum": 0.004745869061305894, "min": 0.004745869061305894}}, "EndTime": 1552273378.554612, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552273378.554605}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.006051138314544855, "sum": 0.006051138314544855, "min": 0.006051138314544855}}, "EndTime": 1552273378.55464, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552273378.554633}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004873958140761409, "sum": 0.004873958140761409, "min": 0.004873958140761409}}, "EndTime": 1552273378.554666, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552273378.554659}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.006153497193401782, "sum": 0.006153497193401782, "min": 0.006153497193401782}}, "EndTime": 1552273378.5547, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552273378.554687}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.006753806056688778, "sum": 0.006753806056688778, "min": 0.006753806056688778}}, "EndTime": 1552273378.554747, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552273378.554734}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005827406466306754, "sum": 0.005827406466306754, "min": 0.005827406466306754}}, "EndTime": 1552273378.554795, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552273378.554781}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0067534985362584865, "sum": 0.0067534985362584865, "min": 0.0067534985362584865}}, "EndTime": 1552273378.554846, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552273378.55483}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005827982135154494, "sum": 0.005827982135154494, "min": 0.005827982135154494}}, "EndTime": 1552273378.554903, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552273378.554887}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005348205740727372, "sum": 0.005348205740727372, "min": 0.005348205740727372}}, "EndTime": 1552273378.554959, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552273378.554943}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005855620055941481, "sum": 0.005855620055941481, "min": 0.005855620055941481}}, "EndTime": 1552273378.555012, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552273378.554997}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005346461805865992, "sum": 0.005346461805865992, "min": 0.005346461805865992}}, "EndTime": 1552273378.555066, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552273378.555051}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005852809699336488, "sum": 0.005852809699336488, "min": 0.005852809699336488}}, "EndTime": 1552273378.555121, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552273378.555106}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012566139340999738, "sum": 0.012566139340999738, "min": 0.012566139340999738}}, "EndTime": 1552273378.555175, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552273378.555161}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012038233971475956, "sum": 0.012038233971475956, "min": 0.012038233971475956}}, "EndTime": 1552273378.55523, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552273378.555215}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01256493431239871, "sum": 0.01256493431239871, "min": 0.01256493431239871}}, "EndTime": 1552273378.555298, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552273378.555283}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012039702561632473, "sum": 0.012039702561632473, "min": 0.012039702561632473}}, "EndTime": 1552273378.555351, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552273378.555337}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0118642488490397, "sum": 0.0118642488490397, "min": 0.0118642488490397}}, "EndTime": 1552273378.555399, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552273378.555386}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011934922299193377, "sum": 0.011934922299193377, "min": 0.011934922299193377}}, "EndTime": 1552273378.55545, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552273378.555436}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011862491466891226, "sum": 0.011862491466891226, "min": 0.011862491466891226}}, "EndTime": 1552273378.555502, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552273378.555487}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011933327715001513, "sum": 0.011933327715001513, "min": 0.011933327715001513}}, "EndTime": 1552273378.555555, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552273378.555541}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.013505783789121924, "sum": 0.013505783789121924, "min": 0.013505783789121924}}, "EndTime": 1552273378.555609, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552273378.555595}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.013001762073842724, "sum": 0.013001762073842724, "min": 0.013001762073842724}}, "EndTime": 1552273378.555664, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552273378.555648}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0135056490993979, "sum": 0.0135056490993979, "min": 0.0135056490993979}}, "EndTime": 1552273378.555719, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552273378.555703}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.013002546663859381, "sum": 0.013002546663859381, "min": 0.013002546663859381}}, "EndTime": 1552273378.555762, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552273378.55575}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01285640471544697, "sum": 0.01285640471544697, "min": 0.01285640471544697}}, "EndTime": 1552273378.555814, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552273378.5558}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012896197460404593, "sum": 0.012896197460404593, "min": 0.012896197460404593}}, "EndTime": 1552273378.555858, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552273378.555844}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012857996290652596, "sum": 0.012857996290652596, "min": 0.012857996290652596}}, "EndTime": 1552273378.55591, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552273378.555896}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012898726200338584, "sum": 0.012898726200338584, "min": 0.012898726200338584}}, "EndTime": 1552273378.555959, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552273378.555949}
    [0m
    [31m[03/11/2019 03:02:58 INFO 140148523243328] #quality_metric: host=algo-1, epoch=5, train binary_classification_cross_entropy_objective <loss>=0.00627634365595[0m
    [31m[03/11/2019 03:02:58 INFO 140148523243328] #early_stopping_criteria_metric: host=algo-1, epoch=5, criteria=binary_classification_cross_entropy_objective, value=0.00474586906131[0m
    [31m[03/11/2019 03:02:58 INFO 140148523243328] Epoch 5: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:02:58 INFO 140148523243328] #progress_metric: host=algo-1, completed 40 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 1212, "sum": 1212.0, "min": 1212}, "Total Records Seen": {"count": 1, "max": 1208184, "sum": 1208184.0, "min": 1208184}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 8, "sum": 8.0, "min": 8}}, "EndTime": 1552273378.55854, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552273372.796999}
    [0m
    [31m[03/11/2019 03:02:58 INFO 140148523243328] #throughput_metric: host=algo-1, train throughput=34601.7684724 records/second[0m
    [31m[2019-03-11 03:02:58.558] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 7, "duration": 5761, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005869037068668922, "sum": 0.005869037068668922, "min": 0.005869037068668922}}, "EndTime": 1552273384.564513, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552273384.564422}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004911643637484642, "sum": 0.004911643637484642, "min": 0.004911643637484642}}, "EndTime": 1552273384.564609, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552273384.564593}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005869176341061616, "sum": 0.005869176341061616, "min": 0.005869176341061616}}, "EndTime": 1552273384.564665, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552273384.564652}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004912145971053809, "sum": 0.004912145971053809, "min": 0.004912145971053809}}, "EndTime": 1552273384.564708, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552273384.564697}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004701146499894972, "sum": 0.004701146499894972, "min": 0.004701146499894972}}, "EndTime": 1552273384.564751, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552273384.56474}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.00605768408989487, "sum": 0.00605768408989487, "min": 0.00605768408989487}}, "EndTime": 1552273384.564902, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552273384.564781}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0049241019635643795, "sum": 0.0049241019635643795, "min": 0.0049241019635643795}}, "EndTime": 1552273384.564961, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552273384.564947}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005674689029628907, "sum": 0.005674689029628907, "min": 0.005674689029628907}}, "EndTime": 1552273384.565003, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552273384.564993}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.006389971089722523, "sum": 0.006389971089722523, "min": 0.006389971089722523}}, "EndTime": 1552273384.565041, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552273384.565032}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.00557527757050404, "sum": 0.00557527757050404, "min": 0.00557527757050404}}, "EndTime": 1552273384.565088, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552273384.565077}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.00638973372306057, "sum": 0.00638973372306057, "min": 0.00638973372306057}}, "EndTime": 1552273384.565127, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552273384.565117}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005575775871324779, "sum": 0.005575775871324779, "min": 0.005575775871324779}}, "EndTime": 1552273384.565174, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552273384.565159}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0053742247679125724, "sum": 0.0053742247679125724, "min": 0.0053742247679125724}}, "EndTime": 1552273384.565244, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552273384.565204}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005830225506020551, "sum": 0.005830225506020551, "min": 0.005830225506020551}}, "EndTime": 1552273384.565305, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552273384.565291}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005373988723635074, "sum": 0.005373988723635074, "min": 0.005373988723635074}}, "EndTime": 1552273384.565345, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552273384.565335}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005828681292246334, "sum": 0.005828681292246334, "min": 0.005828681292246334}}, "EndTime": 1552273384.565383, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552273384.565373}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012353397164512519, "sum": 0.012353397164512519, "min": 0.012353397164512519}}, "EndTime": 1552273384.56542, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552273384.56541}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011946621701945012, "sum": 0.011946621701945012, "min": 0.011946621701945012}}, "EndTime": 1552273384.565457, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552273384.565447}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012352558684708486, "sum": 0.012352558684708486, "min": 0.012352558684708486}}, "EndTime": 1552273384.565494, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552273384.565485}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011947648996084779, "sum": 0.011947648996084779, "min": 0.011947648996084779}}, "EndTime": 1552273384.565531, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552273384.565522}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011843015904402614, "sum": 0.011843015904402614, "min": 0.011843015904402614}}, "EndTime": 1552273384.565568, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552273384.565559}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011943208752565049, "sum": 0.011943208752565049, "min": 0.011943208752565049}}, "EndTime": 1552273384.565616, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552273384.565605}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011843548118768626, "sum": 0.011843548118768626, "min": 0.011843548118768626}}, "EndTime": 1552273384.565658, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552273384.565646}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01194197491904599, "sum": 0.01194197491904599, "min": 0.01194197491904599}}, "EndTime": 1552273384.565709, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552273384.565694}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.013294168243456127, "sum": 0.013294168243456127, "min": 0.013294168243456127}}, "EndTime": 1552273384.565783, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552273384.565771}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012910012778325297, "sum": 0.012910012778325297, "min": 0.012910012778325297}}, "EndTime": 1552273384.565832, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552273384.565818}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.013293834397541218, "sum": 0.013293834397541218, "min": 0.013293834397541218}}, "EndTime": 1552273384.565888, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552273384.565873}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012910769558432114, "sum": 0.012910769558432114, "min": 0.012910769558432114}}, "EndTime": 1552273384.565945, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552273384.565929}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012855722399213206, "sum": 0.012855722399213206, "min": 0.012855722399213206}}, "EndTime": 1552273384.56601, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552273384.565995}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01291352930979513, "sum": 0.01291352930979513, "min": 0.01291352930979513}}, "EndTime": 1552273384.566063, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552273384.566048}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01285700117643155, "sum": 0.01285700117643155, "min": 0.01285700117643155}}, "EndTime": 1552273384.566094, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552273384.566086}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01290991102031727, "sum": 0.01290991102031727, "min": 0.01290991102031727}}, "EndTime": 1552273384.566133, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552273384.56612}
    [0m
    [31m[03/11/2019 03:03:04 INFO 140148523243328] #quality_metric: host=algo-1, epoch=6, train binary_classification_cross_entropy_objective <loss>=0.00586903706867[0m
    [31m[03/11/2019 03:03:04 INFO 140148523243328] #early_stopping_criteria_metric: host=algo-1, epoch=6, criteria=binary_classification_cross_entropy_objective, value=0.00470114649989[0m
    [31m[03/11/2019 03:03:04 INFO 140148523243328] Epoch 6: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:03:04 INFO 140148523243328] #progress_metric: host=algo-1, completed 46 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 1412, "sum": 1412.0, "min": 1412}, "Total Records Seen": {"count": 1, "max": 1407548, "sum": 1407548.0, "min": 1407548}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 9, "sum": 9.0, "min": 9}}, "EndTime": 1552273384.568649, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552273378.55882}
    [0m
    [31m[03/11/2019 03:03:04 INFO 140148523243328] #throughput_metric: host=algo-1, train throughput=33172.3006211 records/second[0m
    [31m[2019-03-11 03:03:04.568] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 8, "duration": 6009, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005591161470317361, "sum": 0.005591161470317361, "min": 0.005591161470317361}}, "EndTime": 1552273390.406994, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552273390.406937}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004708522209570036, "sum": 0.004708522209570036, "min": 0.004708522209570036}}, "EndTime": 1552273390.407065, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552273390.407053}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005591602163698206, "sum": 0.005591602163698206, "min": 0.005591602163698206}}, "EndTime": 1552273390.4071, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552273390.407092}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0047086963965066115, "sum": 0.0047086963965066115, "min": 0.0047086963965066115}}, "EndTime": 1552273390.407133, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552273390.407125}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004754318347378592, "sum": 0.004754318347378592, "min": 0.004754318347378592}}, "EndTime": 1552273390.407169, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552273390.407156}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005659493515764049, "sum": 0.005659493515764049, "min": 0.005659493515764049}}, "EndTime": 1552273390.407215, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552273390.407205}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004616728931216139, "sum": 0.004616728931216139, "min": 0.004616728931216139}}, "EndTime": 1552273390.407243, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552273390.407236}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.00646706811327431, "sum": 0.00646706811327431, "min": 0.00646706811327431}}, "EndTime": 1552273390.40727, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552273390.407263}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.006146658008422084, "sum": 0.006146658008422084, "min": 0.006146658008422084}}, "EndTime": 1552273390.407296, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552273390.407289}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005423311430006171, "sum": 0.005423311430006171, "min": 0.005423311430006171}}, "EndTime": 1552273390.407321, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552273390.407315}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0061464657795489135, "sum": 0.0061464657795489135, "min": 0.0061464657795489135}}, "EndTime": 1552273390.407347, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552273390.40734}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0054237357953085975, "sum": 0.0054237357953085975, "min": 0.0054237357953085975}}, "EndTime": 1552273390.407373, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552273390.407366}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005315854691380831, "sum": 0.005315854691380831, "min": 0.005315854691380831}}, "EndTime": 1552273390.407399, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552273390.407392}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005809691983251716, "sum": 0.005809691983251716, "min": 0.005809691983251716}}, "EndTime": 1552273390.407424, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552273390.407418}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005315677183057795, "sum": 0.005315677183057795, "min": 0.005315677183057795}}, "EndTime": 1552273390.407476, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552273390.407461}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005809011261367319, "sum": 0.005809011261367319, "min": 0.005809011261367319}}, "EndTime": 1552273390.407541, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552273390.407524}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012224241410068531, "sum": 0.012224241410068531, "min": 0.012224241410068531}}, "EndTime": 1552273390.407597, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552273390.407581}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01190716770486017, "sum": 0.01190716770486017, "min": 0.01190716770486017}}, "EndTime": 1552273390.407654, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552273390.407638}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01222359538078308, "sum": 0.01222359538078308, "min": 0.01222359538078308}}, "EndTime": 1552273390.407711, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552273390.407696}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011907873830603594, "sum": 0.011907873830603594, "min": 0.011907873830603594}}, "EndTime": 1552273390.407776, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552273390.407761}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011822155673300201, "sum": 0.011822155673300201, "min": 0.011822155673300201}}, "EndTime": 1552273390.407841, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552273390.407825}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011951955908506959, "sum": 0.011951955908506959, "min": 0.011951955908506959}}, "EndTime": 1552273390.407896, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552273390.407882}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011823138146544222, "sum": 0.011823138146544222, "min": 0.011823138146544222}}, "EndTime": 1552273390.407958, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552273390.407942}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011950859326214048, "sum": 0.011950859326214048, "min": 0.011950859326214048}}, "EndTime": 1552273390.40802, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552273390.408004}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0131656456448924, "sum": 0.0131656456448924, "min": 0.0131656456448924}}, "EndTime": 1552273390.40808, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552273390.408064}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012870260859254617, "sum": 0.012870260859254617, "min": 0.012870260859254617}}, "EndTime": 1552273390.408141, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552273390.408126}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.013165357339322267, "sum": 0.013165357339322267, "min": 0.013165357339322267}}, "EndTime": 1552273390.4082, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552273390.408185}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012870954639348552, "sum": 0.012870954639348552, "min": 0.012870954639348552}}, "EndTime": 1552273390.40826, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552273390.408245}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012855612519398407, "sum": 0.012855612519398407, "min": 0.012855612519398407}}, "EndTime": 1552273390.408319, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552273390.408304}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012923838847845643, "sum": 0.012923838847845643, "min": 0.012923838847845643}}, "EndTime": 1552273390.408385, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552273390.408368}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0128559720863649, "sum": 0.0128559720863649, "min": 0.0128559720863649}}, "EndTime": 1552273390.408447, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552273390.408431}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012921573549658808, "sum": 0.012921573549658808, "min": 0.012921573549658808}}, "EndTime": 1552273390.408507, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552273390.408492}
    [0m
    [31m[03/11/2019 03:03:10 INFO 140148523243328] #quality_metric: host=algo-1, epoch=7, train binary_classification_cross_entropy_objective <loss>=0.00559116147032[0m
    [31m[03/11/2019 03:03:10 INFO 140148523243328] #early_stopping_criteria_metric: host=algo-1, epoch=7, criteria=binary_classification_cross_entropy_objective, value=0.00461672893122[0m
    [31m[03/11/2019 03:03:10 INFO 140148523243328] Epoch 7: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:03:10 INFO 140148523243328] #progress_metric: host=algo-1, completed 53 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 1612, "sum": 1612.0, "min": 1612}, "Total Records Seen": {"count": 1, "max": 1606912, "sum": 1606912.0, "min": 1606912}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 10, "sum": 10.0, "min": 10}}, "EndTime": 1552273390.411019, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552273384.568933}
    [0m
    [31m[03/11/2019 03:03:10 INFO 140148523243328] #throughput_metric: host=algo-1, train throughput=34124.751695 records/second[0m
    [31m[2019-03-11 03:03:10.411] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 9, "duration": 5842, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005391925841719661, "sum": 0.005391925841719661, "min": 0.005391925841719661}}, "EndTime": 1552273396.368999, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552273396.368934}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0045703682456184275, "sum": 0.0045703682456184275, "min": 0.0045703682456184275}}, "EndTime": 1552273396.369074, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552273396.369061}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005392487336642778, "sum": 0.005392487336642778, "min": 0.005392487336642778}}, "EndTime": 1552273396.369128, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552273396.369114}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004570626125263808, "sum": 0.004570626125263808, "min": 0.004570626125263808}}, "EndTime": 1552273396.369177, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552273396.369162}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004525650208619372, "sum": 0.004525650208619372, "min": 0.004525650208619372}}, "EndTime": 1552273396.369212, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552273396.369203}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005934561043453576, "sum": 0.005934561043453576, "min": 0.005934561043453576}}, "EndTime": 1552273396.369241, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552273396.369232}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0046104487279551715, "sum": 0.0046104487279551715, "min": 0.0046104487279551715}}, "EndTime": 1552273396.369291, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552273396.369276}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0064192624762279904, "sum": 0.0064192624762279904, "min": 0.0064192624762279904}}, "EndTime": 1552273396.369342, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552273396.369328}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005975344323632705, "sum": 0.005975344323632705, "min": 0.005975344323632705}}, "EndTime": 1552273396.369396, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552273396.36938}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005327865115362196, "sum": 0.005327865115362196, "min": 0.005327865115362196}}, "EndTime": 1552273396.369451, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552273396.369436}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.00597518141904668, "sum": 0.00597518141904668, "min": 0.00597518141904668}}, "EndTime": 1552273396.369491, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552273396.369482}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005328222943909803, "sum": 0.005328222943909803, "min": 0.005328222943909803}}, "EndTime": 1552273396.369539, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552273396.369526}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005267092317792039, "sum": 0.005267092317792039, "min": 0.005267092317792039}}, "EndTime": 1552273396.369584, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552273396.369572}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0057899654630440564, "sum": 0.0057899654630440564, "min": 0.0057899654630440564}}, "EndTime": 1552273396.369623, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552273396.369609}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005267111534748844, "sum": 0.005267111534748844, "min": 0.005267111534748844}}, "EndTime": 1552273396.369677, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552273396.369662}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005789906225611816, "sum": 0.005789906225611816, "min": 0.005789906225611816}}, "EndTime": 1552273396.369767, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552273396.369749}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012141525111605773, "sum": 0.012141525111605773, "min": 0.012141525111605773}}, "EndTime": 1552273396.369825, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552273396.369809}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011890671925928126, "sum": 0.011890671925928126, "min": 0.011890671925928126}}, "EndTime": 1552273396.369881, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552273396.369865}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012141005372282248, "sum": 0.012141005372282248, "min": 0.012141005372282248}}, "EndTime": 1552273396.369935, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552273396.36992}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011891155160252173, "sum": 0.011891155160252173, "min": 0.011891155160252173}}, "EndTime": 1552273396.369991, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552273396.369976}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011817079468588135, "sum": 0.011817079468588135, "min": 0.011817079468588135}}, "EndTime": 1552273396.370045, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552273396.37003}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011961612669666808, "sum": 0.011961612669666808, "min": 0.011961612669666808}}, "EndTime": 1552273396.370101, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552273396.370086}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011818041877530928, "sum": 0.011818041877530928, "min": 0.011818041877530928}}, "EndTime": 1552273396.370155, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552273396.370139}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011960513301231154, "sum": 0.011960513301231154, "min": 0.011960513301231154}}, "EndTime": 1552273396.370215, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552273396.370199}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01308310868153021, "sum": 0.01308310868153021, "min": 0.01308310868153021}}, "EndTime": 1552273396.370272, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552273396.370257}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012853279242563487, "sum": 0.012853279242563487, "min": 0.012853279242563487}}, "EndTime": 1552273396.370326, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552273396.370311}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.013083011689497598, "sum": 0.013083011689497598, "min": 0.013083011689497598}}, "EndTime": 1552273396.370378, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552273396.370364}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012853929442376946, "sum": 0.012853929442376946, "min": 0.012853929442376946}}, "EndTime": 1552273396.370434, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552273396.370418}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01285552091574549, "sum": 0.01285552091574549, "min": 0.01285552091574549}}, "EndTime": 1552273396.370488, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552273396.370473}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012931668805117582, "sum": 0.012931668805117582, "min": 0.012931668805117582}}, "EndTime": 1552273396.370548, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552273396.370533}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012855314294896534, "sum": 0.012855314294896534, "min": 0.012855314294896534}}, "EndTime": 1552273396.370612, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552273396.370596}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012921399402258983, "sum": 0.012921399402258983, "min": 0.012921399402258983}}, "EndTime": 1552273396.370667, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552273396.370652}
    [0m
    [31m[03/11/2019 03:03:16 INFO 140148523243328] #quality_metric: host=algo-1, epoch=8, train binary_classification_cross_entropy_objective <loss>=0.00539192584172[0m
    [31m[03/11/2019 03:03:16 INFO 140148523243328] #early_stopping_criteria_metric: host=algo-1, epoch=8, criteria=binary_classification_cross_entropy_objective, value=0.00452565020862[0m
    [31m[03/11/2019 03:03:16 INFO 140148523243328] Epoch 8: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:03:16 INFO 140148523243328] #progress_metric: host=algo-1, completed 60 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 1812, "sum": 1812.0, "min": 1812}, "Total Records Seen": {"count": 1, "max": 1806276, "sum": 1806276.0, "min": 1806276}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 11, "sum": 11.0, "min": 11}}, "EndTime": 1552273396.373269, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552273390.411302}
    [0m
    [31m[03/11/2019 03:03:16 INFO 140148523243328] #throughput_metric: host=algo-1, train throughput=33438.5466813 records/second[0m
    [31m[2019-03-11 03:03:16.373] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 10, "duration": 5962, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005243560808387833, "sum": 0.005243560808387833, "min": 0.005243560808387833}}, "EndTime": 1552273402.015322, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552273402.015263}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004472267618430919, "sum": 0.004472267618430919, "min": 0.004472267618430919}}, "EndTime": 1552273402.015397, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552273402.015384}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005244257314121304, "sum": 0.005244257314121304, "min": 0.005244257314121304}}, "EndTime": 1552273402.015448, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552273402.015435}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004471835252927176, "sum": 0.004471835252927176, "min": 0.004471835252927176}}, "EndTime": 1552273402.015493, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552273402.01548}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004443225746628028, "sum": 0.004443225746628028, "min": 0.004443225746628028}}, "EndTime": 1552273402.015525, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552273402.015517}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0056344015265829, "sum": 0.0056344015265829, "min": 0.0056344015265829}}, "EndTime": 1552273402.015552, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552273402.015545}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004439634162876474, "sum": 0.004439634162876474, "min": 0.004439634162876474}}, "EndTime": 1552273402.015588, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552273402.015574}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005886269171857954, "sum": 0.005886269171857954, "min": 0.005886269171857954}}, "EndTime": 1552273402.015635, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552273402.015621}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005849934860689556, "sum": 0.005849934860689556, "min": 0.005849934860689556}}, "EndTime": 1552273402.015682, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552273402.015668}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005266024878875694, "sum": 0.005266024878875694, "min": 0.005266024878875694}}, "EndTime": 1552273402.015731, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552273402.015717}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005849792432545417, "sum": 0.005849792432545417, "min": 0.005849792432545417}}, "EndTime": 1552273402.015782, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552273402.015768}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0052663132647174085, "sum": 0.0052663132647174085, "min": 0.0052663132647174085}}, "EndTime": 1552273402.015835, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552273402.01582}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005242697265279952, "sum": 0.005242697265279952, "min": 0.005242697265279952}}, "EndTime": 1552273402.01589, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552273402.015875}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.00577369510797999, "sum": 0.00577369510797999, "min": 0.00577369510797999}}, "EndTime": 1552273402.015944, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552273402.015929}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005243133715648747, "sum": 0.005243133715648747, "min": 0.005243133715648747}}, "EndTime": 1552273402.01602, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552273402.015994}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005774005268985902, "sum": 0.005774005268985902, "min": 0.005774005268985902}}, "EndTime": 1552273402.016073, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552273402.016058}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012086536772886113, "sum": 0.012086536772886113, "min": 0.012086536772886113}}, "EndTime": 1552273402.016135, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552273402.016119}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011884216188186377, "sum": 0.011884216188186377, "min": 0.011884216188186377}}, "EndTime": 1552273402.0162, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552273402.016181}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012086114896601769, "sum": 0.012086114896601769, "min": 0.012086114896601769}}, "EndTime": 1552273402.016259, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552273402.016242}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011884549309859924, "sum": 0.011884549309859924, "min": 0.011884549309859924}}, "EndTime": 1552273402.01632, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552273402.016305}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011817668989675128, "sum": 0.011817668989675128, "min": 0.011817668989675128}}, "EndTime": 1552273402.016381, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552273402.016366}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011972069397044541, "sum": 0.011972069397044541, "min": 0.011972069397044541}}, "EndTime": 1552273402.016436, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552273402.016421}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011817923034255828, "sum": 0.011817923034255828, "min": 0.011817923034255828}}, "EndTime": 1552273402.016487, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552273402.016473}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011971008896827697, "sum": 0.011971008896827697, "min": 0.011971008896827697}}, "EndTime": 1552273402.016537, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552273402.016523}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.013027740849921452, "sum": 0.013027740849921452, "min": 0.013027740849921452}}, "EndTime": 1552273402.016592, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552273402.016576}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012846210579776285, "sum": 0.012846210579776285, "min": 0.012846210579776285}}, "EndTime": 1552273402.016654, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552273402.016638}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.013027749891856209, "sum": 0.013027749891856209, "min": 0.013027749891856209}}, "EndTime": 1552273402.016717, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552273402.016701}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0128467862899579, "sum": 0.0128467862899579, "min": 0.0128467862899579}}, "EndTime": 1552273402.016775, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552273402.016759}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012855197807053225, "sum": 0.012855197807053225, "min": 0.012855197807053225}}, "EndTime": 1552273402.01683, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552273402.016815}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012939482097649694, "sum": 0.012939482097649694, "min": 0.012939482097649694}}, "EndTime": 1552273402.016892, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552273402.016877}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012855091443612948, "sum": 0.012855091443612948, "min": 0.012855091443612948}}, "EndTime": 1552273402.016955, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552273402.016939}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012936638314520294, "sum": 0.012936638314520294, "min": 0.012936638314520294}}, "EndTime": 1552273402.017007, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552273402.016993}
    [0m
    [31m[03/11/2019 03:03:22 INFO 140148523243328] #quality_metric: host=algo-1, epoch=9, train binary_classification_cross_entropy_objective <loss>=0.00524356080839[0m
    [31m[03/11/2019 03:03:22 INFO 140148523243328] #early_stopping_criteria_metric: host=algo-1, epoch=9, criteria=binary_classification_cross_entropy_objective, value=0.00443963416288[0m
    [31m[03/11/2019 03:03:22 INFO 140148523243328] Epoch 9: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:03:22 INFO 140148523243328] #progress_metric: host=algo-1, completed 66 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 2012, "sum": 2012.0, "min": 2012}, "Total Records Seen": {"count": 1, "max": 2005640, "sum": 2005640.0, "min": 2005640}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 12, "sum": 12.0, "min": 12}}, "EndTime": 1552273402.019597, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552273396.373555}
    [0m
    [31m[03/11/2019 03:03:22 INFO 140148523243328] #throughput_metric: host=algo-1, train throughput=35309.6311814 records/second[0m
    [31m[2019-03-11 03:03:22.019] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 11, "duration": 5646, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005129939455482828, "sum": 0.005129939455482828, "min": 0.005129939455482828}}, "EndTime": 1552273407.801985, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552273407.801927}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004400380541631325, "sum": 0.004400380541631325, "min": 0.004400380541631325}}, "EndTime": 1552273407.802057, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552273407.802044}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005130695801284445, "sum": 0.005130695801284445, "min": 0.005130695801284445}}, "EndTime": 1552273407.802111, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552273407.802096}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004400535478364283, "sum": 0.004400535478364283, "min": 0.004400535478364283}}, "EndTime": 1552273407.802165, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552273407.802151}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0044496707453500085, "sum": 0.0044496707453500085, "min": 0.0044496707453500085}}, "EndTime": 1552273407.802221, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552273407.802207}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.006058443752964537, "sum": 0.006058443752964537, "min": 0.006058443752964537}}, "EndTime": 1552273407.802277, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552273407.802262}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004425278927513103, "sum": 0.004425278927513103, "min": 0.004425278927513103}}, "EndTime": 1552273407.802341, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552273407.802325}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.00605832583618224, "sum": 0.00605832583618224, "min": 0.00605832583618224}}, "EndTime": 1552273407.802407, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552273407.80239}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005755311613106847, "sum": 0.005755311613106847, "min": 0.005755311613106847}}, "EndTime": 1552273407.802469, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552273407.802453}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005224952253864039, "sum": 0.005224952253864039, "min": 0.005224952253864039}}, "EndTime": 1552273407.802531, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552273407.802516}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005755185417793504, "sum": 0.005755185417793504, "min": 0.005755185417793504}}, "EndTime": 1552273407.802593, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552273407.802578}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0052251979810508655, "sum": 0.0052251979810508655, "min": 0.0052251979810508655}}, "EndTime": 1552273407.802645, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552273407.802632}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005231819331645965, "sum": 0.005231819331645965, "min": 0.005231819331645965}}, "EndTime": 1552273407.802692, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552273407.802679}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005756311737412784, "sum": 0.005756311737412784, "min": 0.005756311737412784}}, "EndTime": 1552273407.802743, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552273407.802729}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005232467777465456, "sum": 0.005232467777465456, "min": 0.005232467777465456}}, "EndTime": 1552273407.802793, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552273407.802779}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005756838018870234, "sum": 0.005756838018870234, "min": 0.005756838018870234}}, "EndTime": 1552273407.802841, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552273407.802828}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01204894584387391, "sum": 0.01204894584387391, "min": 0.01204894584387391}}, "EndTime": 1552273407.802897, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552273407.802884}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011882044413580966, "sum": 0.011882044413580966, "min": 0.011882044413580966}}, "EndTime": 1552273407.802943, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552273407.80293}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012048594359776482, "sum": 0.012048594359776482, "min": 0.012048594359776482}}, "EndTime": 1552273407.802993, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552273407.802979}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011882273570377024, "sum": 0.011882273570377024, "min": 0.011882273570377024}}, "EndTime": 1552273407.803045, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552273407.80303}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011827303311932626, "sum": 0.011827303311932626, "min": 0.011827303311932626}}, "EndTime": 1552273407.803098, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552273407.803082}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011982332607010501, "sum": 0.011982332607010501, "min": 0.011982332607010501}}, "EndTime": 1552273407.803154, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552273407.803139}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011827303381421458, "sum": 0.011827303381421458, "min": 0.011827303381421458}}, "EndTime": 1552273407.803208, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552273407.803192}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011981394795317147, "sum": 0.011981394795317147, "min": 0.011981394795317147}}, "EndTime": 1552273407.803253, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552273407.80324}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012989261237820189, "sum": 0.012989261237820189, "min": 0.012989261237820189}}, "EndTime": 1552273407.803321, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552273407.803306}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012843442471782167, "sum": 0.012843442471782167, "min": 0.012843442471782167}}, "EndTime": 1552273407.803364, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552273407.803355}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012989297969856454, "sum": 0.012989297969856454, "min": 0.012989297969856454}}, "EndTime": 1552273407.803405, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552273407.803391}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012843923484260713, "sum": 0.012843923484260713, "min": 0.012843923484260713}}, "EndTime": 1552273407.803456, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552273407.803444}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012854552010195938, "sum": 0.012854552010195938, "min": 0.012854552010195938}}, "EndTime": 1552273407.803513, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552273407.803497}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012953675210775442, "sum": 0.012953675210775442, "min": 0.012953675210775442}}, "EndTime": 1552273407.803576, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552273407.80356}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01285420082262413, "sum": 0.01285420082262413, "min": 0.01285420082262413}}, "EndTime": 1552273407.803694, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552273407.803648}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012952155484626042, "sum": 0.012952155484626042, "min": 0.012952155484626042}}, "EndTime": 1552273407.80375, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552273407.803734}
    [0m
    [31m[03/11/2019 03:03:27 INFO 140148523243328] #quality_metric: host=algo-1, epoch=10, train binary_classification_cross_entropy_objective <loss>=0.00512993945548[0m
    [31m[03/11/2019 03:03:27 INFO 140148523243328] #early_stopping_criteria_metric: host=algo-1, epoch=10, criteria=binary_classification_cross_entropy_objective, value=0.00440038054163[0m
    [31m[03/11/2019 03:03:27 INFO 140148523243328] Epoch 10: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:03:27 INFO 140148523243328] #progress_metric: host=algo-1, completed 73 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 2212, "sum": 2212.0, "min": 2212}, "Total Records Seen": {"count": 1, "max": 2205004, "sum": 2205004.0, "min": 2205004}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 13, "sum": 13.0, "min": 13}}, "EndTime": 1552273407.806474, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552273402.019863}
    [0m
    [31m[03/11/2019 03:03:27 INFO 140148523243328] #throughput_metric: host=algo-1, train throughput=34451.9622639 records/second[0m
    [31m[2019-03-11 03:03:27.806] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 12, "duration": 5786, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0050408838328404645, "sum": 0.0050408838328404645, "min": 0.0050408838328404645}}, "EndTime": 1552273413.86107, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552273413.86101}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004347342520802464, "sum": 0.004347342520802464, "min": 0.004347342520802464}}, "EndTime": 1552273413.861143, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552273413.86113}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005041672517905882, "sum": 0.005041672517905882, "min": 0.005041672517905882}}, "EndTime": 1552273413.861196, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552273413.861182}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004347448597601311, "sum": 0.004347448597601311, "min": 0.004347448597601311}}, "EndTime": 1552273413.861248, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552273413.861234}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004311184193021688, "sum": 0.004311184193021688, "min": 0.004311184193021688}}, "EndTime": 1552273413.861305, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552273413.861291}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0057367004239229705, "sum": 0.0057367004239229705, "min": 0.0057367004239229705}}, "EndTime": 1552273413.861357, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552273413.861343}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004303851319617362, "sum": 0.004303851319617362, "min": 0.004303851319617362}}, "EndTime": 1552273413.861412, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552273413.861396}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005506682152349745, "sum": 0.005506682152349745, "min": 0.005506682152349745}}, "EndTime": 1552273413.861467, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552273413.861452}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005682184379304474, "sum": 0.005682184379304474, "min": 0.005682184379304474}}, "EndTime": 1552273413.861522, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552273413.861507}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005197159426895219, "sum": 0.005197159426895219, "min": 0.005197159426895219}}, "EndTime": 1552273413.861587, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552273413.86157}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005682071232316482, "sum": 0.005682071232316482, "min": 0.005682071232316482}}, "EndTime": 1552273413.861643, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552273413.861628}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005197354152274491, "sum": 0.005197354152274491, "min": 0.005197354152274491}}, "EndTime": 1552273413.861699, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552273413.861684}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0052181945592913795, "sum": 0.0052181945592913795, "min": 0.0052181945592913795}}, "EndTime": 1552273413.861783, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552273413.861766}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005725392285902895, "sum": 0.005725392285902895, "min": 0.005725392285902895}}, "EndTime": 1552273413.861841, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552273413.861825}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005218749440794614, "sum": 0.005218749440794614, "min": 0.005218749440794614}}, "EndTime": 1552273413.861901, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552273413.861885}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005725960174397608, "sum": 0.005725960174397608, "min": 0.005725960174397608}}, "EndTime": 1552273413.861954, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552273413.86194}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012022629658780506, "sum": 0.012022629658780506, "min": 0.012022629658780506}}, "EndTime": 1552273413.862012, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552273413.861996}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011881658866177851, "sum": 0.011881658866177851, "min": 0.011881658866177851}}, "EndTime": 1552273413.862068, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552273413.862052}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012022314350808685, "sum": 0.012022314350808685, "min": 0.012022314350808685}}, "EndTime": 1552273413.862123, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552273413.862108}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011881814344444467, "sum": 0.011881814344444467, "min": 0.011881814344444467}}, "EndTime": 1552273413.862178, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552273413.862163}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011839835171723486, "sum": 0.011839835171723486, "min": 0.011839835171723486}}, "EndTime": 1552273413.862234, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552273413.862218}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011991599743090683, "sum": 0.011991599743090683, "min": 0.011991599743090683}}, "EndTime": 1552273413.86229, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552273413.862275}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011839938913757477, "sum": 0.011839938913757477, "min": 0.011839938913757477}}, "EndTime": 1552273413.862343, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552273413.862329}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011990834723165885, "sum": 0.011990834723165885, "min": 0.011990834723165885}}, "EndTime": 1552273413.862407, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552273413.86239}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012961713818449471, "sum": 0.012961713818449471, "min": 0.012961713818449471}}, "EndTime": 1552273413.862469, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552273413.862453}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01284264255169049, "sum": 0.01284264255169049, "min": 0.01284264255169049}}, "EndTime": 1552273413.862529, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552273413.862513}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012961710998161353, "sum": 0.012961710998161353, "min": 0.012961710998161353}}, "EndTime": 1552273413.862583, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552273413.862568}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012842846664352033, "sum": 0.012842846664352033, "min": 0.012842846664352033}}, "EndTime": 1552273413.862633, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552273413.862618}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012853404417109849, "sum": 0.012853404417109849, "min": 0.012853404417109849}}, "EndTime": 1552273413.862686, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552273413.862671}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012962785278732453, "sum": 0.012962785278732453, "min": 0.012962785278732453}}, "EndTime": 1552273413.862735, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552273413.86272}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012852913996682095, "sum": 0.012852913996682095, "min": 0.012852913996682095}}, "EndTime": 1552273413.86278, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552273413.862766}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012966532389722278, "sum": 0.012966532389722278, "min": 0.012966532389722278}}, "EndTime": 1552273413.862831, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552273413.862816}
    [0m
    [31m[03/11/2019 03:03:33 INFO 140148523243328] #quality_metric: host=algo-1, epoch=11, train binary_classification_cross_entropy_objective <loss>=0.00504088383284[0m
    [31m[03/11/2019 03:03:33 INFO 140148523243328] #early_stopping_criteria_metric: host=algo-1, epoch=11, criteria=binary_classification_cross_entropy_objective, value=0.00430385131962[0m
    [31m[03/11/2019 03:03:33 INFO 140148523243328] Epoch 11: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:03:33 INFO 140148523243328] #progress_metric: host=algo-1, completed 80 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 2412, "sum": 2412.0, "min": 2412}, "Total Records Seen": {"count": 1, "max": 2404368, "sum": 2404368.0, "min": 2404368}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 14, "sum": 14.0, "min": 14}}, "EndTime": 1552273413.865293, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552273407.806738}
    [0m
    [31m[03/11/2019 03:03:33 INFO 140148523243328] #throughput_metric: host=algo-1, train throughput=32905.5779206 records/second[0m
    [31m[2019-03-11 03:03:33.865] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 13, "duration": 6058, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004969843264201179, "sum": 0.004969843264201179, "min": 0.004969843264201179}}, "EndTime": 1552273419.94294, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552273419.94286}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004306342939935138, "sum": 0.004306342939935138, "min": 0.004306342939935138}}, "EndTime": 1552273419.943023, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552273419.943005}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004970650205660106, "sum": 0.004970650205660106, "min": 0.004970650205660106}}, "EndTime": 1552273419.943079, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552273419.943064}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004306516928888445, "sum": 0.004306516928888445, "min": 0.004306516928888445}}, "EndTime": 1552273419.94313, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552273419.943116}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.00428055602671513, "sum": 0.00428055602671513, "min": 0.00428055602671513}}, "EndTime": 1552273419.943179, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552273419.943166}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.006004417111451302, "sum": 0.006004417111451302, "min": 0.006004417111451302}}, "EndTime": 1552273419.943224, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552273419.943211}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004273409339501031, "sum": 0.004273409339501031, "min": 0.004273409339501031}}, "EndTime": 1552273419.943278, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552273419.943262}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005807543110158575, "sum": 0.005807543110158575, "min": 0.005807543110158575}}, "EndTime": 1552273419.943329, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552273419.943316}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005624567857938795, "sum": 0.005624567857938795, "min": 0.005624567857938795}}, "EndTime": 1552273419.943381, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552273419.943366}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005178005310758274, "sum": 0.005178005310758274, "min": 0.005178005310758274}}, "EndTime": 1552273419.943436, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552273419.94342}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005624464228524635, "sum": 0.005624464228524635, "min": 0.005624464228524635}}, "EndTime": 1552273419.943491, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552273419.943476}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005178166065683317, "sum": 0.005178166065683317, "min": 0.005178166065683317}}, "EndTime": 1552273419.943543, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552273419.943529}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005200647512272974, "sum": 0.005200647512272974, "min": 0.005200647512272974}}, "EndTime": 1552273419.943595, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552273419.943581}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0057072182270749726, "sum": 0.0057072182270749726, "min": 0.0057072182270749726}}, "EndTime": 1552273419.943648, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552273419.943633}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005200967681767353, "sum": 0.005200967681767353, "min": 0.005200967681767353}}, "EndTime": 1552273419.943702, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552273419.943687}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005707736464002025, "sum": 0.005707736464002025, "min": 0.005707736464002025}}, "EndTime": 1552273419.943766, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552273419.943751}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012003761139347325, "sum": 0.012003761139347325, "min": 0.012003761139347325}}, "EndTime": 1552273419.943818, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552273419.943803}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011881993036174295, "sum": 0.011881993036174295, "min": 0.011881993036174295}}, "EndTime": 1552273419.943869, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552273419.943855}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012003464668839421, "sum": 0.012003464668839421, "min": 0.012003464668839421}}, "EndTime": 1552273419.94392, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552273419.943905}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011882093836913755, "sum": 0.011882093836913755, "min": 0.011882093836913755}}, "EndTime": 1552273419.943972, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552273419.943958}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011850788091295328, "sum": 0.011850788091295328, "min": 0.011850788091295328}}, "EndTime": 1552273419.944026, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552273419.944011}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011999677057841315, "sum": 0.011999677057841315, "min": 0.011999677057841315}}, "EndTime": 1552273419.944082, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552273419.944067}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011850821938946019, "sum": 0.011850821938946019, "min": 0.011850821938946019}}, "EndTime": 1552273419.944144, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552273419.944129}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011999078996217431, "sum": 0.011999078996217431, "min": 0.011999078996217431}}, "EndTime": 1552273419.944201, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552273419.944186}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012941368530743087, "sum": 0.012941368530743087, "min": 0.012941368530743087}}, "EndTime": 1552273419.944265, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552273419.944249}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0128426011369456, "sum": 0.0128426011369456, "min": 0.0128426011369456}}, "EndTime": 1552273419.94432, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552273419.944306}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012941372148954688, "sum": 0.012941372148954688, "min": 0.012941372148954688}}, "EndTime": 1552273419.944394, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552273419.944377}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012842444234157927, "sum": 0.012842444234157927, "min": 0.012842444234157927}}, "EndTime": 1552273419.944457, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552273419.944441}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012852127319604308, "sum": 0.012852127319604308, "min": 0.012852127319604308}}, "EndTime": 1552273419.94452, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552273419.944504}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012978015047221927, "sum": 0.012978015047221927, "min": 0.012978015047221927}}, "EndTime": 1552273419.944575, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552273419.94456}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012851788702921653, "sum": 0.012851788702921653, "min": 0.012851788702921653}}, "EndTime": 1552273419.94464, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552273419.944622}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012976963284626677, "sum": 0.012976963284626677, "min": 0.012976963284626677}}, "EndTime": 1552273419.944698, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552273419.944683}
    [0m
    [31m[03/11/2019 03:03:39 INFO 140148523243328] #quality_metric: host=algo-1, epoch=12, train binary_classification_cross_entropy_objective <loss>=0.0049698432642[0m
    [31m[03/11/2019 03:03:39 INFO 140148523243328] #early_stopping_criteria_metric: host=algo-1, epoch=12, criteria=binary_classification_cross_entropy_objective, value=0.0042734093395[0m
    [31m[03/11/2019 03:03:39 INFO 140148523243328] Epoch 12: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:03:39 INFO 140148523243328] #progress_metric: host=algo-1, completed 86 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 2612, "sum": 2612.0, "min": 2612}, "Total Records Seen": {"count": 1, "max": 2603732, "sum": 2603732.0, "min": 2603732}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 15, "sum": 15.0, "min": 15}}, "EndTime": 1552273419.947241, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552273413.865574}
    [0m
    [31m[03/11/2019 03:03:39 INFO 140148523243328] #throughput_metric: host=algo-1, train throughput=32780.4963269 records/second[0m
    [31m[2019-03-11 03:03:39.947] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 14, "duration": 6081, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004912319385825689, "sum": 0.004912319385825689, "min": 0.004912319385825689}}, "EndTime": 1552273425.94075, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552273425.94069}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004274392560199278, "sum": 0.004274392560199278, "min": 0.004274392560199278}}, "EndTime": 1552273425.940835, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552273425.940817}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004913153540548967, "sum": 0.004913153540548967, "min": 0.004913153540548967}}, "EndTime": 1552273425.94089, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552273425.940876}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.00427452319710698, "sum": 0.00427452319710698, "min": 0.00427452319710698}}, "EndTime": 1552273425.940946, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552273425.940932}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004213615064195652, "sum": 0.004213615064195652, "min": 0.004213615064195652}}, "EndTime": 1552273425.940995, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552273425.940982}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.006460462951480444, "sum": 0.006460462951480444, "min": 0.006460462951480444}}, "EndTime": 1552273425.941042, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552273425.941029}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004210366838840983, "sum": 0.004210366838840983, "min": 0.004210366838840983}}, "EndTime": 1552273425.941097, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552273425.941082}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005640471553607801, "sum": 0.005640471553607801, "min": 0.005640471553607801}}, "EndTime": 1552273425.941152, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552273425.941136}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0055784549036217695, "sum": 0.0055784549036217695, "min": 0.0055784549036217695}}, "EndTime": 1552273425.941208, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552273425.941192}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0051646082239534385, "sum": 0.0051646082239534385, "min": 0.0051646082239534385}}, "EndTime": 1552273425.941264, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552273425.941248}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005578360546174361, "sum": 0.005578360546174361, "min": 0.005578360546174361}}, "EndTime": 1552273425.941318, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552273425.941303}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0051647417164927155, "sum": 0.0051647417164927155, "min": 0.0051647417164927155}}, "EndTime": 1552273425.941372, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552273425.941357}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0051832432740896795, "sum": 0.0051832432740896795, "min": 0.0051832432740896795}}, "EndTime": 1552273425.941427, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552273425.941411}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005687902828556808, "sum": 0.005687902828556808, "min": 0.005687902828556808}}, "EndTime": 1552273425.941485, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552273425.94147}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005183386637337843, "sum": 0.005183386637337843, "min": 0.005183386637337843}}, "EndTime": 1552273425.941545, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552273425.941525}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005688361336537941, "sum": 0.005688361336537941, "min": 0.005688361336537941}}, "EndTime": 1552273425.941603, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552273425.941587}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011989877118537174, "sum": 0.011989877118537174, "min": 0.011989877118537174}}, "EndTime": 1552273425.941668, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552273425.941651}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011882599697640195, "sum": 0.011882599697640195, "min": 0.011882599697640195}}, "EndTime": 1552273425.941751, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552273425.941714}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011989592566562058, "sum": 0.011989592566562058, "min": 0.011989592566562058}}, "EndTime": 1552273425.941808, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552273425.941792}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01188265965691763, "sum": 0.01188265965691763, "min": 0.01188265965691763}}, "EndTime": 1552273425.941864, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552273425.941848}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011865591880065113, "sum": 0.011865591880065113, "min": 0.011865591880065113}}, "EndTime": 1552273425.94192, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552273425.941904}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012006625869765353, "sum": 0.012006625869765353, "min": 0.012006625869765353}}, "EndTime": 1552273425.941976, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552273425.94196}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011865580839727392, "sum": 0.011865580839727392, "min": 0.011865580839727392}}, "EndTime": 1552273425.942031, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552273425.942016}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012006160519230905, "sum": 0.012006160519230905, "min": 0.012006160519230905}}, "EndTime": 1552273425.942086, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552273425.942071}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012926021113467577, "sum": 0.012926021113467577, "min": 0.012926021113467577}}, "EndTime": 1552273425.942143, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552273425.942131}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012842781799522477, "sum": 0.012842781799522477, "min": 0.012842781799522477}}, "EndTime": 1552273425.942209, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552273425.942191}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01292604173008521, "sum": 0.01292604173008521, "min": 0.01292604173008521}}, "EndTime": 1552273425.942275, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552273425.942258}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012842306256294251, "sum": 0.012842306256294251, "min": 0.012842306256294251}}, "EndTime": 1552273425.942339, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552273425.942323}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012850760931345686, "sum": 0.012850760931345686, "min": 0.012850760931345686}}, "EndTime": 1552273425.942395, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552273425.942379}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01298285104581459, "sum": 0.01298285104581459, "min": 0.01298285104581459}}, "EndTime": 1552273425.94245, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552273425.942435}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012851034403446331, "sum": 0.012851034403446331, "min": 0.012851034403446331}}, "EndTime": 1552273425.942505, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552273425.942489}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01298556806693724, "sum": 0.01298556806693724, "min": 0.01298556806693724}}, "EndTime": 1552273425.942562, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552273425.942547}
    [0m
    [31m[03/11/2019 03:03:45 INFO 140148523243328] #quality_metric: host=algo-1, epoch=13, train binary_classification_cross_entropy_objective <loss>=0.00491231938583[0m
    [31m[03/11/2019 03:03:45 INFO 140148523243328] #early_stopping_criteria_metric: host=algo-1, epoch=13, criteria=binary_classification_cross_entropy_objective, value=0.00421036683884[0m
    [31m[03/11/2019 03:03:45 INFO 140148523243328] Epoch 13: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:03:45 INFO 140148523243328] #progress_metric: host=algo-1, completed 93 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 2812, "sum": 2812.0, "min": 2812}, "Total Records Seen": {"count": 1, "max": 2803096, "sum": 2803096.0, "min": 2803096}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 16, "sum": 16.0, "min": 16}}, "EndTime": 1552273425.945116, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552273419.947515}
    [0m
    [31m[03/11/2019 03:03:45 INFO 140148523243328] #throughput_metric: host=algo-1, train throughput=33239.9763946 records/second[0m
    [31m[2019-03-11 03:03:45.945] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 15, "duration": 5997, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0048651198454238665, "sum": 0.0048651198454238665, "min": 0.0048651198454238665}}, "EndTime": 1552273432.029888, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552273432.029825}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004249175175650036, "sum": 0.004249175175650036, "min": 0.004249175175650036}}, "EndTime": 1552273432.029963, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552273432.029951}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004865984468004811, "sum": 0.004865984468004811, "min": 0.004865984468004811}}, "EndTime": 1552273432.030018, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552273432.030004}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004249360133056066, "sum": 0.004249360133056066, "min": 0.004249360133056066}}, "EndTime": 1552273432.030054, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552273432.030043}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004188439261673683, "sum": 0.004188439261673683, "min": 0.004188439261673683}}, "EndTime": 1552273432.0301, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552273432.03009}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005883642438293701, "sum": 0.005883642438293701, "min": 0.005883642438293701}}, "EndTime": 1552273432.030129, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552273432.030122}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004183407771078187, "sum": 0.004183407771078187, "min": 0.004183407771078187}}, "EndTime": 1552273432.030173, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552273432.03016}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.006083663204656774, "sum": 0.006083663204656774, "min": 0.006083663204656774}}, "EndTime": 1552273432.030218, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552273432.030205}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005541083724654499, "sum": 0.005541083724654499, "min": 0.005541083724654499}}, "EndTime": 1552273432.030266, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552273432.030253}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005155116175886374, "sum": 0.005155116175886374, "min": 0.005155116175886374}}, "EndTime": 1552273432.030312, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552273432.030302}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.00554099817491656, "sum": 0.00554099817491656, "min": 0.00554099817491656}}, "EndTime": 1552273432.030341, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552273432.030333}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005155223439386742, "sum": 0.005155223439386742, "min": 0.005155223439386742}}, "EndTime": 1552273432.030367, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552273432.03036}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005167508804318893, "sum": 0.005167508804318893, "min": 0.005167508804318893}}, "EndTime": 1552273432.030392, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552273432.030386}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005668988515384233, "sum": 0.005668988515384233, "min": 0.005668988515384233}}, "EndTime": 1552273432.030418, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552273432.030411}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005167576271385404, "sum": 0.005167576271385404, "min": 0.005167576271385404}}, "EndTime": 1552273432.030448, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552273432.030436}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005669342463040471, "sum": 0.005669342463040471, "min": 0.005669342463040471}}, "EndTime": 1552273432.030494, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552273432.030484}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011979322374765598, "sum": 0.011979322374765598, "min": 0.011979322374765598}}, "EndTime": 1552273432.030522, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552273432.030516}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011883317816197572, "sum": 0.011883317816197572, "min": 0.011883317816197572}}, "EndTime": 1552273432.030563, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552273432.03055}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011979054548033517, "sum": 0.011979054548033517, "min": 0.011979054548033517}}, "EndTime": 1552273432.030619, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552273432.03061}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011883347391483173, "sum": 0.011883347391483173, "min": 0.011883347391483173}}, "EndTime": 1552273432.030646, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552273432.030639}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011879505603157696, "sum": 0.011879505603157696, "min": 0.011879505603157696}}, "EndTime": 1552273432.030671, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552273432.030664}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012012594955650406, "sum": 0.012012594955650406, "min": 0.012012594955650406}}, "EndTime": 1552273432.030696, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552273432.030689}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011879369377490864, "sum": 0.011879369377490864, "min": 0.011879369377490864}}, "EndTime": 1552273432.030741, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552273432.030728}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012012267659057925, "sum": 0.012012267659057925, "min": 0.012012267659057925}}, "EndTime": 1552273432.030777, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552273432.030769}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012914269195729165, "sum": 0.012914269195729165, "min": 0.012914269195729165}}, "EndTime": 1552273432.030803, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552273432.030796}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012842971407588402, "sum": 0.012842971407588402, "min": 0.012842971407588402}}, "EndTime": 1552273432.030828, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552273432.030821}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012914280618255463, "sum": 0.012914280618255463, "min": 0.012914280618255463}}, "EndTime": 1552273432.030852, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552273432.030846}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012842341140886048, "sum": 0.012842341140886048, "min": 0.012842341140886048}}, "EndTime": 1552273432.030876, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552273432.03087}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01284956756249145, "sum": 0.01284956756249145, "min": 0.01284956756249145}}, "EndTime": 1552273432.030917, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552273432.030904}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012990787114929314, "sum": 0.012990787114929314, "min": 0.012990787114929314}}, "EndTime": 1552273432.030946, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552273432.030939}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012850412979796904, "sum": 0.012850412979796904, "min": 0.012850412979796904}}, "EndTime": 1552273432.030971, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552273432.030965}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012991039510348334, "sum": 0.012991039510348334, "min": 0.012991039510348334}}, "EndTime": 1552273432.030995, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552273432.030989}
    [0m
    [31m[03/11/2019 03:03:52 INFO 140148523243328] #quality_metric: host=algo-1, epoch=14, train binary_classification_cross_entropy_objective <loss>=0.00486511984542[0m
    [31m[03/11/2019 03:03:52 INFO 140148523243328] #early_stopping_criteria_metric: host=algo-1, epoch=14, criteria=binary_classification_cross_entropy_objective, value=0.00418340777108[0m
    [31m[03/11/2019 03:03:52 INFO 140148523243328] Epoch 14: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:03:52 INFO 140148523243328] #progress_metric: host=algo-1, completed 100 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 3012, "sum": 3012.0, "min": 3012}, "Total Records Seen": {"count": 1, "max": 3002460, "sum": 3002460.0, "min": 3002460}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 17, "sum": 17.0, "min": 17}}, "EndTime": 1552273432.033596, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552273425.94538}
    [0m
    [31m[03/11/2019 03:03:52 INFO 140148523243328] #throughput_metric: host=algo-1, train throughput=32745.2797839 records/second[0m
    [31m[03/11/2019 03:03:52 WARNING 140148523243328] wait_for_all_workers will not sync workers since the kv store is not running distributed[0m
    [31m[03/11/2019 03:03:52 WARNING 140148523243328] wait_for_all_workers will not sync workers since the kv store is not running distributed[0m
    [31m[2019-03-11 03:03:52.034] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 16, "duration": 6088, "num_examples": 200}[0m
    [31m[2019-03-11 03:03:52.039] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 17, "duration": 5, "num_examples": 1}[0m
    [31m[2019-03-11 03:03:52.787] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 18, "duration": 745, "num_examples": 200}[0m
    [31m[03/11/2019 03:03:53 INFO 140148523243328] #train_score (algo-1) : ('binary_classification_cross_entropy_objective', 0.0041373788497844092)[0m
    [31m[03/11/2019 03:03:53 INFO 140148523243328] #train_score (algo-1) : ('binary_classification_accuracy', 0.99930278284946128)[0m
    [31m[03/11/2019 03:03:53 INFO 140148523243328] #train_score (algo-1) : ('binary_f_1.000', 0.8005738880918221)[0m
    [31m[03/11/2019 03:03:53 INFO 140148523243328] #train_score (algo-1) : ('precision', 0.8063583815028902)[0m
    [31m[03/11/2019 03:03:53 INFO 140148523243328] #train_score (algo-1) : ('recall', 0.7948717948717948)[0m
    [31m[03/11/2019 03:03:53 INFO 140148523243328] #quality_metric: host=algo-1, train binary_classification_cross_entropy_objective <loss>=0.00413737884978[0m
    [31m[03/11/2019 03:03:53 INFO 140148523243328] #quality_metric: host=algo-1, train binary_classification_accuracy <score>=0.999302782849[0m
    [31m[03/11/2019 03:03:53 INFO 140148523243328] #quality_metric: host=algo-1, train binary_f_1.000 <score>=0.800573888092[0m
    [31m[03/11/2019 03:03:53 INFO 140148523243328] #quality_metric: host=algo-1, train precision <score>=0.806358381503[0m
    [31m[03/11/2019 03:03:53 INFO 140148523243328] #quality_metric: host=algo-1, train recall <score>=0.794871794872[0m
    [31m[03/11/2019 03:03:53 INFO 140148523243328] Best model found for hyperparameters: {"lr_scheduler_step": 10, "wd": 0.0001, "optimizer": "adam", "lr_scheduler_factor": 0.99, "l1": 0.0, "learning_rate": 0.1, "lr_scheduler_minimum_lr": 0.0001}[0m
    [31m[03/11/2019 03:03:53 INFO 140148523243328] Saved checkpoint to "/tmp/tmphihGE0/mx-mod-0000.params"[0m
    [31m[03/11/2019 03:03:53 INFO 140148523243328] Test data is not provided.[0m
    [31m[2019-03-11 03:03:53.484] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 19, "duration": 696, "num_examples": 200}[0m
    [31m[2019-03-11 03:03:53.484] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "duration": 90914, "num_epochs": 20, "num_examples": 3413}[0m
    [31m#metrics {"Metrics": {"totaltime": {"count": 1, "max": 91130.3060054779, "sum": 91130.3060054779, "min": 91130.3060054779}, "finalize.time": {"count": 1, "max": 1443.4418678283691, "sum": 1443.4418678283691, "min": 1443.4418678283691}, "initialize.time": {"count": 1, "max": 194.02790069580078, "sum": 194.02790069580078, "min": 194.02790069580078}, "check_early_stopping.time": {"count": 15, "max": 0.9090900421142578, "sum": 12.160062789916992, "min": 0.7579326629638672}, "setuptime": {"count": 1, "max": 15.455961227416992, "sum": 15.455961227416992, "min": 15.455961227416992}, "update.time": {"count": 15, "max": 6237.394094467163, "sum": 89385.01119613647, "min": 5645.884037017822}, "epochs": {"count": 1, "max": 15, "sum": 15.0, "min": 15}}, "EndTime": 1552273433.484398, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1552273342.442778}
    [0m
    
    2019-03-11 03:04:04 Uploading - Uploading generated training model
    2019-03-11 03:04:04 Completed - Training job completed
    Billable seconds: 149
    CPU times: user 613 ms, sys: 40.2 ms, total: 653 ms
    Wall time: 4min 42s


### EXERCISE: Deploy the trained model

Deploy your model to create a predictor. We'll use this to make predictions on our test data and evaluate the model.


```python
%%time 
# deploy and create a predictor
linear_predictor = linear.deploy(initial_instance_count=1, instance_type='ml.t2.medium')
```

    INFO:sagemaker:Creating model with name: linear-learner-2019-03-11-03-04-22-561
    INFO:sagemaker:Creating endpoint with name linear-learner-2019-03-11-02-59-39-742


    ----------------------------------------------------------------------------------------!CPU times: user 437 ms, sys: 25 ms, total: 462 ms
    Wall time: 7min 25s


---
# Evaluating Your Model

Once your model is deployed, you can see how it performs when applied to the test data.

According to the deployed [predictor documentation](https://sagemaker.readthedocs.io/en/stable/linear_learner.html#sagemaker.LinearLearnerPredictor), this predictor expects an `ndarray` of input features and returns a list of Records.
> "The prediction is stored in the "predicted_label" key of the `Record.label` field."

Let's first test our model on just one test point, to see the resulting list.


```python
# test one prediction
test_x_np = test_features.astype('float32')
result = linear_predictor.predict(test_x_np[0])

print(result)
```

    [label {
      key: "predicted_label"
      value {
        float32_tensor {
          values: 0.0
        }
      }
    }
    label {
      key: "score"
      value {
        float32_tensor {
          values: 0.001805478474125266
        }
      }
    }
    ]


### Helper function for evaluation


The provided function below, takes in a deployed predictor, some test features and labels, and returns a dictionary of metrics; calculating false negatives and positives as well as recall, precision, and accuracy.


```python
# code to evaluate the endpoint on test data
# returns a variety of model metrics
def evaluate(predictor, test_features, test_labels, verbose=True):
    """
    Evaluate a model on a test set given the prediction endpoint.  
    Return binary classification metrics.
    :param predictor: A prediction endpoint
    :param test_features: Test features
    :param test_labels: Class labels for test data
    :param verbose: If True, prints a table of all performance metrics
    :return: A dictionary of performance metrics.
    """
    
    # We have a lot of test data, so we'll split it into batches of 100
    # split the test data set into batches and evaluate using prediction endpoint    
    prediction_batches = [predictor.predict(batch) for batch in np.array_split(test_features, 100)]
    
    # LinearLearner produces a `predicted_label` for each data point in a batch
    # get the 'predicted_label' for every point in a batch
    test_preds = np.concatenate([np.array([x.label['predicted_label'].float32_tensor.values[0] for x in batch]) 
                                 for batch in prediction_batches])
    
    # calculate true positives, false positives, true negatives, false negatives
    tp = np.logical_and(test_labels, test_preds).sum()
    fp = np.logical_and(1-test_labels, test_preds).sum()
    tn = np.logical_and(1-test_labels, 1-test_preds).sum()
    fn = np.logical_and(test_labels, 1-test_preds).sum()
    
    # calculate binary classification metrics
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    
    # printing a table of metrics
    if verbose:
        print(pd.crosstab(test_labels, test_preds, rownames=['actual (row)'], colnames=['prediction (col)']))
        print("\n{:<11} {:.3f}".format('Recall:', recall))
        print("{:<11} {:.3f}".format('Precision:', precision))
        print("{:<11} {:.3f}".format('Accuracy:', accuracy))
        print()
        
    return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn, 
            'Precision': precision, 'Recall': recall, 'Accuracy': accuracy}

```

### Test Results

The cell below runs the `evaluate` function. 

The code assumes that you have a defined `predictor` and `test_features` and `test_labels` from previously-run cells.


```python
print('Metrics for simple, LinearLearner.\n')

# get metrics for linear predictor
metrics = evaluate(linear_predictor, 
                   test_features.astype('float32'), 
                   test_labels, 
                   verbose=True) # verbose means we'll print out the metrics

```

    Metrics for simple, LinearLearner.
    
    prediction (col)    0.0  1.0
    actual (row)                
    0.0               85269   33
    1.0                  32  109
    
    Recall:     0.773
    Precision:  0.768
    Accuracy:   0.999
    


We can see that this model gets a very high accuracy of 99.9% ! But it still misclassifies about 30 (each) of our valid and fraudulent cases, which results in much lower values for recall and precision.

Next, let's delete this endpoint and discuss ways to improve this model.

## Delete the Endpoint

I've added a convenience function to delete prediction endpoints after we're done with them. And if you're done evaluating the model, you should delete your model endpoint!


```python
# Deletes a precictor.endpoint
def delete_endpoint(predictor):
        try:
            boto3.client('sagemaker').delete_endpoint(EndpointName=predictor.endpoint)
            print('Deleted {}'.format(predictor.endpoint))
        except:
            print('Already deleted: {}'.format(predictor.endpoint))
```


```python
# delete the predictor endpoint 
delete_endpoint(linear_predictor)
```

    Deleted linear-learner-2019-03-11-02-59-39-742


---

# Model Improvements

The default LinearLearner got a high accuracy, but still classified fraudulent and valid data points incorrectly. Specifically classifying more than 30 points as false negatives (incorrectly labeled, fraudulent transactions), and a little over 30 points as false positives (incorrectly labeled, valid transactions). Let's think about what, during training, could cause this behavior and what we could improve.

**1. Model optimization**
* If we imagine that we are designing this model for use in a bank application, we know that users do *not* want any valid transactions to be categorized as fraudulent. That is, we want to have as few **false positives** (0s classified as 1s) as possible. 
* On the other hand, if our bank manager asks for an application that will catch almost *all* cases of fraud, even if it means a higher number of false positives, then we'd want as few **false negatives** as possible.
* To train according to specific product demands and goals, we do not want to optimize for accuracy only. Instead, we want to optimize for a metric that can help us decrease the number of false positives or negatives. 

<img src='notebook_ims/precision_recall.png' width=40% />
     
In this notebook, we'll look at different cases for tuning a model and make an optimization decision, accordingly.

**2. Imbalanced training data**
* At the start of this notebook, we saw that only about 0.17% of the data was labeled as fraudulent. So, even if a model labels **all** of our data as valid, it will still have a high accuracy. 
* This may result in some overfitting towards valid data, which accounts for some **false negatives**; cases in which fraudulent data (1) is incorrectly characterized as valid (0).

So, let's address these issues in order; first, tuning our model and optimizing for a specific metric during training, and second, accounting for class imbalance in the training set. 


## Improvement: Model Tuning

Optimizing according to a specific metric is called **model tuning**, and SageMaker provides a number of ways to automatically tune a model.


### Create a LinearLearner and tune for higher precision 

**Scenario:**
* A bank has asked you to build a model that detects cases of fraud with an accuracy of about 85%. 

In this case, we want to build a model that has as many true positives and as few false negatives, as possible. This corresponds to a model with a high **recall**: true positives / (true positives + false negatives). 

To aim for a specific metric, LinearLearner offers the hyperparameter `binary_classifier_model_selection_criteria`, which is the model evaluation criteria for the training dataset. A reference to this parameter is in [LinearLearner's documentation](https://sagemaker.readthedocs.io/en/stable/linear_learner.html#sagemaker.LinearLearner). We'll also have to further specify the exact value we want to aim for; read more about the details of the parameters, [here](https://docs.aws.amazon.com/sagemaker/latest/dg/ll_hyperparameters.html).

I will assume that performance on a training set will be within about 5% of the performance on a test set. So, for a recall of about 85%, I'll aim for a bit higher, 90%.


```python
# instantiate a LinearLearner
# tune the model for a higher recall
linear_recall = LinearLearner(role=role,
                              train_instance_count=1, 
                              train_instance_type='ml.c4.xlarge',
                              predictor_type='binary_classifier',
                              output_path=output_path,
                              sagemaker_session=sagemaker_session,
                              epochs=15,
                              binary_classifier_model_selection_criteria='precision_at_target_recall', # target recall
                              target_recall=0.9) # 90% recall

```

### Train the tuned estimator

Fit the new, tuned estimator on the formatted training data.


```python
%%time 
# train the estimator on formatted training data
linear_recall.fit(formatted_train_data)
```

    INFO:sagemaker:Creating training-job with name: linear-learner-2019-03-11-03-12-00-255


    2019-03-11 03:12:00 Starting - Starting the training job...
    2019-03-11 03:12:07 Starting - Launching requested ML instances......
    2019-03-11 03:13:10 Starting - Preparing the instances for training.........
    2019-03-11 03:14:49 Downloading - Downloading input data
    2019-03-11 03:14:49 Training - Training image download completed. Training in progress.
    [31mDocker entrypoint called with argument(s): train[0m
    [31m[03/11/2019 03:14:51 INFO 140227862767424] Reading default configuration from /opt/amazon/lib/python2.7/site-packages/algorithm/default-input.json: {u'loss_insensitivity': u'0.01', u'epochs': u'15', u'init_bias': u'0.0', u'lr_scheduler_factor': u'auto', u'num_calibration_samples': u'10000000', u'accuracy_top_k': u'3', u'_num_kv_servers': u'auto', u'use_bias': u'true', u'num_point_for_scaler': u'10000', u'_log_level': u'info', u'quantile': u'0.5', u'bias_lr_mult': u'auto', u'lr_scheduler_step': u'auto', u'init_method': u'uniform', u'init_sigma': u'0.01', u'lr_scheduler_minimum_lr': u'auto', u'target_recall': u'0.8', u'num_models': u'auto', u'early_stopping_patience': u'3', u'momentum': u'auto', u'unbias_label': u'auto', u'wd': u'auto', u'optimizer': u'auto', u'_tuning_objective_metric': u'', u'early_stopping_tolerance': u'0.001', u'learning_rate': u'auto', u'_kvstore': u'auto', u'normalize_data': u'true', u'binary_classifier_model_selection_criteria': u'accuracy', u'use_lr_scheduler': u'true', u'target_precision': u'0.8', u'unbias_data': u'auto', u'init_scale': u'0.07', u'bias_wd_mult': u'auto', u'f_beta': u'1.0', u'mini_batch_size': u'1000', u'huber_delta': u'1.0', u'num_classes': u'1', u'beta_1': u'auto', u'loss': u'auto', u'beta_2': u'auto', u'_enable_profiler': u'false', u'normalize_label': u'auto', u'_num_gpus': u'auto', u'balance_multiclass_weights': u'false', u'positive_example_weight_mult': u'1.0', u'l1': u'auto', u'margin': u'1.0'}[0m
    [31m[03/11/2019 03:14:51 INFO 140227862767424] Reading provided configuration from /opt/ml/input/config/hyperparameters.json: {u'predictor_type': u'binary_classifier', u'binary_classifier_model_selection_criteria': u'precision_at_target_recall', u'epochs': u'15', u'feature_dim': u'30', u'target_recall': u'0.9', u'mini_batch_size': u'1000'}[0m
    [31m[03/11/2019 03:14:51 INFO 140227862767424] Final configuration: {u'loss_insensitivity': u'0.01', u'epochs': u'15', u'feature_dim': u'30', u'init_bias': u'0.0', u'lr_scheduler_factor': u'auto', u'num_calibration_samples': u'10000000', u'accuracy_top_k': u'3', u'_num_kv_servers': u'auto', u'use_bias': u'true', u'num_point_for_scaler': u'10000', u'_log_level': u'info', u'quantile': u'0.5', u'bias_lr_mult': u'auto', u'lr_scheduler_step': u'auto', u'init_method': u'uniform', u'init_sigma': u'0.01', u'lr_scheduler_minimum_lr': u'auto', u'target_recall': u'0.9', u'num_models': u'auto', u'early_stopping_patience': u'3', u'momentum': u'auto', u'unbias_label': u'auto', u'wd': u'auto', u'optimizer': u'auto', u'_tuning_objective_metric': u'', u'early_stopping_tolerance': u'0.001', u'learning_rate': u'auto', u'_kvstore': u'auto', u'normalize_data': u'true', u'binary_classifier_model_selection_criteria': u'precision_at_target_recall', u'use_lr_scheduler': u'true', u'target_precision': u'0.8', u'unbias_data': u'auto', u'init_scale': u'0.07', u'bias_wd_mult': u'auto', u'f_beta': u'1.0', u'mini_batch_size': u'1000', u'huber_delta': u'1.0', u'num_classes': u'1', u'predictor_type': u'binary_classifier', u'beta_1': u'auto', u'loss': u'auto', u'beta_2': u'auto', u'_enable_profiler': u'false', u'normalize_label': u'auto', u'_num_gpus': u'auto', u'balance_multiclass_weights': u'false', u'positive_example_weight_mult': u'1.0', u'l1': u'auto', u'margin': u'1.0'}[0m
    [31m[03/11/2019 03:14:51 WARNING 140227862767424] Loggers have already been setup.[0m
    [31mProcess 1 is a worker.[0m
    [31m[03/11/2019 03:14:51 INFO 140227862767424] Using default worker.[0m
    [31m[2019-03-11 03:14:51.128] [tensorio] [info] batch={"data_pipeline": "/opt/ml/input/data/train", "num_examples": 1000, "features": [{"name": "label_values", "shape": [1], "storage_type": "dense"}, {"name": "values", "shape": [30], "storage_type": "dense"}]}[0m
    [31m[2019-03-11 03:14:51.155] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 0, "duration": 27, "num_examples": 1}[0m
    [31m[03/11/2019 03:14:51 INFO 140227862767424] Create Store: local[0m
    [31m[2019-03-11 03:14:51.199] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 1, "duration": 43, "num_examples": 11}[0m
    [31m[03/11/2019 03:14:51 INFO 140227862767424] Scaler algorithm parameters
     <algorithm.scaler.ScalerAlgorithmStable object at 0x7f88fbc3aa50>[0m
    [31m[03/11/2019 03:14:51 INFO 140227862767424] Scaling model computed with parameters:
     {'stdev_weight': [0m
    [31m[  4.75497891e+04   2.01225400e+00   1.72936726e+00   1.48752689e+00
       1.41830683e+00   1.42959750e+00   1.34760964e+00   1.27067423e+00
       1.24293745e+00   1.09265101e+00   1.05321789e+00   1.01260686e+00
       9.87991810e-01   1.00782645e+00   9.47202206e-01   9.02963459e-01
       8.68877888e-01   8.27179432e-01   8.36477458e-01   8.07050884e-01
       8.00110519e-01   7.55493522e-01   7.21427202e-01   6.25614405e-01
       6.10876381e-01   5.16283095e-01   4.88118291e-01   4.35698181e-01
       3.69419903e-01   2.47155548e+02][0m
    [31m<NDArray 30 @cpu(0)>, 'stdev_label': None, 'mean_label': None, 'mean_weight': [0m
    [31m[  9.44802812e+04  -1.04726264e-02  -1.43008800e-02   1.28451567e-02
       1.87512934e-02  -2.48281248e-02   5.86199807e-03  -7.13069551e-03
      -7.39883492e-03   1.20382467e-02   6.10911567e-03  -3.16866231e-03
       8.64854374e-04   2.46435311e-03   1.56665407e-02   1.12619074e-02
      -4.91584092e-03  -1.56447978e-03   2.45723873e-03   2.82235094e-04
      -3.25949211e-03   6.57527940e-03   3.11945518e-03   6.22356636e-03
      -6.13171898e-04  -3.88089707e-03   1.16021503e-02  -3.21021304e-03
      -5.27510792e-03   8.94287567e+01][0m
    [31m<NDArray 30 @cpu(0)>}[0m
    [31m[03/11/2019 03:14:51 INFO 140227862767424] nvidia-smi took: 0.0251998901367 secs to identify 0 gpus[0m
    [31m[03/11/2019 03:14:51 INFO 140227862767424] Number of GPUs being used: 0[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 11, "sum": 11.0, "min": 11}, "Number of Batches Since Last Reset": {"count": 1, "max": 0, "sum": 0.0, "min": 0}, "Number of Records Since Last Reset": {"count": 1, "max": 0, "sum": 0.0, "min": 0}, "Total Batches Seen": {"count": 1, "max": 12, "sum": 12.0, "min": 12}, "Total Records Seen": {"count": 1, "max": 12000, "sum": 12000.0, "min": 12000}, "Max Records Seen Between Resets": {"count": 1, "max": 11000, "sum": 11000.0, "min": 11000}, "Reset Count": {"count": 1, "max": 2, "sum": 2.0, "min": 2}}, "EndTime": 1552274091.30252, "Dimensions": {"Host": "algo-1", "Meta": "init_train_data_iter", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1552274091.302485}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.12043031547776419, "sum": 0.12043031547776419, "min": 0.12043031547776419}}, "EndTime": 1552274097.5614, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274097.561343}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.11774842056197737, "sum": 0.11774842056197737, "min": 0.11774842056197737}}, "EndTime": 1552274097.561485, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274097.561467}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.12036449890520105, "sum": 0.12036449890520105, "min": 0.12036449890520105}}, "EndTime": 1552274097.561546, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274097.561531}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.11797112150527723, "sum": 0.11797112150527723, "min": 0.11797112150527723}}, "EndTime": 1552274097.561584, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274097.561572}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012310507309758783, "sum": 0.012310507309758783, "min": 0.012310507309758783}}, "EndTime": 1552274097.561635, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274097.561621}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012591959382159327, "sum": 0.012591959382159327, "min": 0.012591959382159327}}, "EndTime": 1552274097.561688, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274097.561672}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012322717984342695, "sum": 0.012322717984342695, "min": 0.012322717984342695}}, "EndTime": 1552274097.561735, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274097.56172}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012404795770743984, "sum": 0.012404795770743984, "min": 0.012404795770743984}}, "EndTime": 1552274097.561778, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274097.561769}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.12042285891393921, "sum": 0.12042285891393921, "min": 0.12042285891393921}}, "EndTime": 1552274097.561813, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274097.5618}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1179497050855627, "sum": 0.1179497050855627, "min": 0.1179497050855627}}, "EndTime": 1552274097.561863, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274097.561849}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.12053752546454195, "sum": 0.12053752546454195, "min": 0.12053752546454195}}, "EndTime": 1552274097.561919, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274097.561904}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.11788512564424294, "sum": 0.11788512564424294, "min": 0.11788512564424294}}, "EndTime": 1552274097.561974, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274097.561959}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012449912267788571, "sum": 0.012449912267788571, "min": 0.012449912267788571}}, "EndTime": 1552274097.562031, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274097.562015}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01263913904151051, "sum": 0.01263913904151051, "min": 0.01263913904151051}}, "EndTime": 1552274097.562079, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274097.562065}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012591561175283775, "sum": 0.012591561175283775, "min": 0.012591561175283775}}, "EndTime": 1552274097.562142, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274097.562127}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012598671647153758, "sum": 0.012598671647153758, "min": 0.012598671647153758}}, "EndTime": 1552274097.562188, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274097.562173}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1229229508883989, "sum": 0.1229229508883989, "min": 0.1229229508883989}}, "EndTime": 1552274097.562249, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274097.562233}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1202098897617666, "sum": 0.1202098897617666, "min": 0.1202098897617666}}, "EndTime": 1552274097.562297, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274097.562282}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.12281814978709772, "sum": 0.12281814978709772, "min": 0.12281814978709772}}, "EndTime": 1552274097.562341, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274097.562327}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1203641751639208, "sum": 0.1203641751639208, "min": 0.1203641751639208}}, "EndTime": 1552274097.562393, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274097.56238}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.019713981734822743, "sum": 0.019713981734822743, "min": 0.019713981734822743}}, "EndTime": 1552274097.562446, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274097.562431}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.019672431648339157, "sum": 0.019672431648339157, "min": 0.019672431648339157}}, "EndTime": 1552274097.562504, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274097.562489}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.019612770365298394, "sum": 0.019612770365298394, "min": 0.019612770365298394}}, "EndTime": 1552274097.562548, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274097.562533}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.019638996485002974, "sum": 0.019638996485002974, "min": 0.019638996485002974}}, "EndTime": 1552274097.562601, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274097.562586}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1237417889312284, "sum": 0.1237417889312284, "min": 0.1237417889312284}}, "EndTime": 1552274097.562656, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274097.56264}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.12108165967164926, "sum": 0.12108165967164926, "min": 0.12108165967164926}}, "EndTime": 1552274097.562708, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274097.562694}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.12376727658180735, "sum": 0.12376727658180735, "min": 0.12376727658180735}}, "EndTime": 1552274097.562752, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274097.562738}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.12122506574889523, "sum": 0.12122506574889523, "min": 0.12122506574889523}}, "EndTime": 1552274097.562804, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274097.562789}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.020484194981991947, "sum": 0.020484194981991947, "min": 0.020484194981991947}}, "EndTime": 1552274097.562857, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274097.562841}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.020433710982302327, "sum": 0.020433710982302327, "min": 0.020433710982302327}}, "EndTime": 1552274097.562917, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274097.562902}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.020589645515434706, "sum": 0.020589645515434706, "min": 0.020589645515434706}}, "EndTime": 1552274097.562974, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274097.562959}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.020449347862346687, "sum": 0.020449347862346687, "min": 0.020449347862346687}}, "EndTime": 1552274097.563028, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274097.563014}
    [0m
    [31m[03/11/2019 03:14:57 INFO 140227862767424] #quality_metric: host=algo-1, epoch=0, train binary_classification_cross_entropy_objective <loss>=0.120430315478[0m
    [31m[03/11/2019 03:14:57 INFO 140227862767424] #early_stopping_criteria_metric: host=algo-1, epoch=0, criteria=binary_classification_cross_entropy_objective, value=0.0123105073098[0m
    [31m[03/11/2019 03:14:57 INFO 140227862767424] Epoch 0: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:14:57 INFO 140227862767424] #progress_metric: host=algo-1, completed 6 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 212, "sum": 212.0, "min": 212}, "Total Records Seen": {"count": 1, "max": 211364, "sum": 211364.0, "min": 211364}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 3, "sum": 3.0, "min": 3}}, "EndTime": 1552274097.566082, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274091.302703}
    [0m
    [31m[03/11/2019 03:14:57 INFO 140227862767424] #throughput_metric: host=algo-1, train throughput=31829.5090282 records/second[0m
    [31m[2019-03-11 03:14:57.566] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 2, "duration": 6263, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01697396398668912, "sum": 0.01697396398668912, "min": 0.01697396398668912}}, "EndTime": 1552274103.950328, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274103.950261}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.014988560247660881, "sum": 0.014988560247660881, "min": 0.014988560247660881}}, "EndTime": 1552274103.950436, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274103.950416}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.016968431084599327, "sum": 0.016968431084599327, "min": 0.016968431084599327}}, "EndTime": 1552274103.950496, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274103.95048}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.014997809695239043, "sum": 0.014997809695239043, "min": 0.014997809695239043}}, "EndTime": 1552274103.95055, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274103.950535}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005413321109123566, "sum": 0.005413321109123566, "min": 0.005413321109123566}}, "EndTime": 1552274103.950597, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274103.950585}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.006144253869377189, "sum": 0.006144253869377189, "min": 0.006144253869377189}}, "EndTime": 1552274103.950649, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274103.950634}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005583528396097859, "sum": 0.005583528396097859, "min": 0.005583528396097859}}, "EndTime": 1552274103.950703, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274103.950687}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005664390350780895, "sum": 0.005664390350780895, "min": 0.005664390350780895}}, "EndTime": 1552274103.950752, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274103.950739}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.017126370190376012, "sum": 0.017126370190376012, "min": 0.017126370190376012}}, "EndTime": 1552274103.950802, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274103.950788}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.015161259157573758, "sum": 0.015161259157573758, "min": 0.015161259157573758}}, "EndTime": 1552274103.950859, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274103.950843}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.017126536081783737, "sum": 0.017126536081783737, "min": 0.017126536081783737}}, "EndTime": 1552274103.950915, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274103.9509}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.015162341254440385, "sum": 0.015162341254440385, "min": 0.015162341254440385}}, "EndTime": 1552274103.950969, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274103.950953}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005557677248015476, "sum": 0.005557677248015476, "min": 0.005557677248015476}}, "EndTime": 1552274103.951036, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274103.95102}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005842554873258026, "sum": 0.005842554873258026, "min": 0.005842554873258026}}, "EndTime": 1552274103.951089, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274103.951075}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005539924988195524, "sum": 0.005539924988195524, "min": 0.005539924988195524}}, "EndTime": 1552274103.951141, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274103.951127}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005890578771356362, "sum": 0.005890578771356362, "min": 0.005890578771356362}}, "EndTime": 1552274103.951194, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274103.951179}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.021226986760470138, "sum": 0.021226986760470138, "min": 0.021226986760470138}}, "EndTime": 1552274103.951246, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274103.951232}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.019429727343458627, "sum": 0.019429727343458627, "min": 0.019429727343458627}}, "EndTime": 1552274103.951301, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274103.951285}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.02121443661253656, "sum": 0.02121443661253656, "min": 0.02121443661253656}}, "EndTime": 1552274103.95135, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274103.951336}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0194394117863334, "sum": 0.0194394117863334, "min": 0.0194394117863334}}, "EndTime": 1552274103.951398, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274103.951385}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011853274800669607, "sum": 0.011853274800669607, "min": 0.011853274800669607}}, "EndTime": 1552274103.951482, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274103.951465}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011885794957678521, "sum": 0.011885794957678521, "min": 0.011885794957678521}}, "EndTime": 1552274103.951539, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274103.951523}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011870770789870066, "sum": 0.011870770789870066, "min": 0.011870770789870066}}, "EndTime": 1552274103.951593, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274103.951578}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011879496791254935, "sum": 0.011879496791254935, "min": 0.011879496791254935}}, "EndTime": 1552274103.951648, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274103.951632}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.02215121567309202, "sum": 0.02215121567309202, "min": 0.02215121567309202}}, "EndTime": 1552274103.951701, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274103.951686}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.020372232250232793, "sum": 0.020372232250232793, "min": 0.020372232250232793}}, "EndTime": 1552274103.951753, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274103.951738}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.02215141379294084, "sum": 0.02215141379294084, "min": 0.02215141379294084}}, "EndTime": 1552274103.951806, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274103.951791}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.02038332456080758, "sum": 0.02038332456080758, "min": 0.02038332456080758}}, "EndTime": 1552274103.951857, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274103.951843}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012853264628343247, "sum": 0.012853264628343247, "min": 0.012853264628343247}}, "EndTime": 1552274103.951909, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274103.951894}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012862902701200551, "sum": 0.012862902701200551, "min": 0.012862902701200551}}, "EndTime": 1552274103.951962, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274103.951947}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012854538045935894, "sum": 0.012854538045935894, "min": 0.012854538045935894}}, "EndTime": 1552274103.952012, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274103.951998}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012862664958939481, "sum": 0.012862664958939481, "min": 0.012862664958939481}}, "EndTime": 1552274103.952066, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274103.95205}
    [0m
    [31m[03/11/2019 03:15:03 INFO 140227862767424] #quality_metric: host=algo-1, epoch=1, train binary_classification_cross_entropy_objective <loss>=0.0169739639867[0m
    [31m[03/11/2019 03:15:03 INFO 140227862767424] #early_stopping_criteria_metric: host=algo-1, epoch=1, criteria=binary_classification_cross_entropy_objective, value=0.00541332110912[0m
    [31m[03/11/2019 03:15:03 INFO 140227862767424] Epoch 1: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:15:03 INFO 140227862767424] #progress_metric: host=algo-1, completed 13 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 412, "sum": 412.0, "min": 412}, "Total Records Seen": {"count": 1, "max": 410728, "sum": 410728.0, "min": 410728}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 4, "sum": 4.0, "min": 4}}, "EndTime": 1552274103.954845, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274097.56637}
    [0m
    [31m[03/11/2019 03:15:03 INFO 140227862767424] #throughput_metric: host=algo-1, train throughput=31206.1674312 records/second[0m
    [31m[2019-03-11 03:15:03.955] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 3, "duration": 6388, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.010330928617985404, "sum": 0.010330928617985404, "min": 0.010330928617985404}}, "EndTime": 1552274109.974779, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274109.974707}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.008765228896883864, "sum": 0.008765228896883864, "min": 0.008765228896883864}}, "EndTime": 1552274109.974866, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274109.974848}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.010327125379188576, "sum": 0.010327125379188576, "min": 0.010327125379188576}}, "EndTime": 1552274109.974957, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274109.974936}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.008771104021887084, "sum": 0.008771104021887084, "min": 0.008771104021887084}}, "EndTime": 1552274109.975027, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274109.975011}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005136206356724303, "sum": 0.005136206356724303, "min": 0.005136206356724303}}, "EndTime": 1552274109.975107, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274109.975064}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005824846665388975, "sum": 0.005824846665388975, "min": 0.005824846665388975}}, "EndTime": 1552274109.975172, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274109.975155}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005184625139787569, "sum": 0.005184625139787569, "min": 0.005184625139787569}}, "EndTime": 1552274109.975231, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274109.975215}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005772609489721869, "sum": 0.005772609489721869, "min": 0.005772609489721869}}, "EndTime": 1552274109.975287, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274109.975272}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.010579428452343198, "sum": 0.010579428452343198, "min": 0.010579428452343198}}, "EndTime": 1552274109.975351, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274109.975335}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.009065929326579798, "sum": 0.009065929326579798, "min": 0.009065929326579798}}, "EndTime": 1552274109.975441, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274109.975397}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.010578739535269426, "sum": 0.010578739535269426, "min": 0.010578739535269426}}, "EndTime": 1552274109.975497, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274109.975481}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.009066744816363157, "sum": 0.009066744816363157, "min": 0.009066744816363157}}, "EndTime": 1552274109.975553, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274109.975538}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005415446985606572, "sum": 0.005415446985606572, "min": 0.005415446985606572}}, "EndTime": 1552274109.975608, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274109.975593}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005902450287761401, "sum": 0.005902450287761401, "min": 0.005902450287761401}}, "EndTime": 1552274109.975662, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274109.975646}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005396837090727073, "sum": 0.005396837090727073, "min": 0.005396837090727073}}, "EndTime": 1552274109.975717, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274109.975701}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005898476717460095, "sum": 0.005898476717460095, "min": 0.005898476717460095}}, "EndTime": 1552274109.975773, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274109.975758}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01543734490332292, "sum": 0.01543734490332292, "min": 0.01543734490332292}}, "EndTime": 1552274109.975827, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274109.975813}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.014191929726145375, "sum": 0.014191929726145375, "min": 0.014191929726145375}}, "EndTime": 1552274109.975884, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274109.975869}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.015432025216931674, "sum": 0.015432025216931674, "min": 0.015432025216931674}}, "EndTime": 1552274109.975939, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274109.975924}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.014196535096096634, "sum": 0.014196535096096634, "min": 0.014196535096096634}}, "EndTime": 1552274109.976001, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274109.975985}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011893838788995791, "sum": 0.011893838788995791, "min": 0.011893838788995791}}, "EndTime": 1552274109.976066, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274109.976049}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011907273306918504, "sum": 0.011907273306918504, "min": 0.011907273306918504}}, "EndTime": 1552274109.976123, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274109.976107}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011899466008397202, "sum": 0.011899466008397202, "min": 0.011899466008397202}}, "EndTime": 1552274109.976187, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274109.976171}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011901343090450345, "sum": 0.011901343090450345, "min": 0.011901343090450345}}, "EndTime": 1552274109.976252, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274109.976235}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.016371653041647907, "sum": 0.016371653041647907, "min": 0.016371653041647907}}, "EndTime": 1552274109.976313, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274109.976297}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.015148403155743777, "sum": 0.015148403155743777, "min": 0.015148403155743777}}, "EndTime": 1552274109.976377, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274109.976361}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.016371930594420315, "sum": 0.016371930594420315, "min": 0.016371930594420315}}, "EndTime": 1552274109.976441, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274109.976425}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01515225470605208, "sum": 0.01515225470605208, "min": 0.01515225470605208}}, "EndTime": 1552274109.976505, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274109.976489}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012857517996625085, "sum": 0.012857517996625085, "min": 0.012857517996625085}}, "EndTime": 1552274109.976569, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274109.976553}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012874755165085721, "sum": 0.012874755165085721, "min": 0.012874755165085721}}, "EndTime": 1552274109.976634, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274109.976617}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012858834195975681, "sum": 0.012858834195975681, "min": 0.012858834195975681}}, "EndTime": 1552274109.976698, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274109.976682}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012874105859042412, "sum": 0.012874105859042412, "min": 0.012874105859042412}}, "EndTime": 1552274109.976756, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274109.97674}
    [0m
    [31m[03/11/2019 03:15:09 INFO 140227862767424] #quality_metric: host=algo-1, epoch=2, train binary_classification_cross_entropy_objective <loss>=0.010330928618[0m
    [31m[03/11/2019 03:15:09 INFO 140227862767424] #early_stopping_criteria_metric: host=algo-1, epoch=2, criteria=binary_classification_cross_entropy_objective, value=0.00513620635672[0m
    [31m[03/11/2019 03:15:09 INFO 140227862767424] Epoch 2: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:15:09 INFO 140227862767424] #progress_metric: host=algo-1, completed 20 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 612, "sum": 612.0, "min": 612}, "Total Records Seen": {"count": 1, "max": 610092, "sum": 610092.0, "min": 610092}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 5, "sum": 5.0, "min": 5}}, "EndTime": 1552274109.979356, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274103.955135}
    [0m
    [31m[03/11/2019 03:15:09 INFO 140227862767424] #throughput_metric: host=algo-1, train throughput=33093.0089148 records/second[0m
    [31m[2019-03-11 03:15:09.979] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 4, "duration": 6024, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.008034130657138537, "sum": 0.008034130657138537, "min": 0.008034130657138537}}, "EndTime": 1552274116.01905, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274116.018989}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.006699231958868516, "sum": 0.006699231958868516, "min": 0.006699231958868516}}, "EndTime": 1552274116.019153, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274116.019133}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.008031785320396998, "sum": 0.008031785320396998, "min": 0.008031785320396998}}, "EndTime": 1552274116.019225, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274116.019207}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.006702904769523659, "sum": 0.006702904769523659, "min": 0.006702904769523659}}, "EndTime": 1552274116.019288, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274116.019271}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0050371206441716335, "sum": 0.0050371206441716335, "min": 0.0050371206441716335}}, "EndTime": 1552274116.019347, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274116.019331}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.006105847063301197, "sum": 0.006105847063301197, "min": 0.006105847063301197}}, "EndTime": 1552274116.019397, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274116.019387}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005057589868504797, "sum": 0.005057589868504797, "min": 0.005057589868504797}}, "EndTime": 1552274116.019492, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274116.019474}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.006192843649405331, "sum": 0.006192843649405331, "min": 0.006192843649405331}}, "EndTime": 1552274116.019554, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274116.019536}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.008379334282036403, "sum": 0.008379334282036403, "min": 0.008379334282036403}}, "EndTime": 1552274116.019611, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274116.019595}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.007128520195208602, "sum": 0.007128520195208602, "min": 0.007128520195208602}}, "EndTime": 1552274116.019667, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274116.019651}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.008378759724410934, "sum": 0.008378759724410934, "min": 0.008378759724410934}}, "EndTime": 1552274116.019725, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274116.019709}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0071292423375287845, "sum": 0.0071292423375287845, "min": 0.0071292423375287845}}, "EndTime": 1552274116.019782, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274116.019767}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005350434124469757, "sum": 0.005350434124469757, "min": 0.005350434124469757}}, "EndTime": 1552274116.01984, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274116.019825}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005897634128230301, "sum": 0.005897634128230301, "min": 0.005897634128230301}}, "EndTime": 1552274116.019895, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274116.019879}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005347798913267989, "sum": 0.005347798913267989, "min": 0.005347798913267989}}, "EndTime": 1552274116.019953, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274116.019937}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.00589653688789013, "sum": 0.00589653688789013, "min": 0.00589653688789013}}, "EndTime": 1552274116.020018, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274116.020003}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.013688772067352755, "sum": 0.013688772067352755, "min": 0.013688772067352755}}, "EndTime": 1552274116.020072, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274116.020057}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012770949257079082, "sum": 0.012770949257079082, "min": 0.012770949257079082}}, "EndTime": 1552274116.02014, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274116.020123}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.013685778299168725, "sum": 0.013685778299168725, "min": 0.013685778299168725}}, "EndTime": 1552274116.020196, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274116.020181}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012773823243289737, "sum": 0.012773823243289737, "min": 0.012773823243289737}}, "EndTime": 1552274116.020254, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274116.020238}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011883480945424218, "sum": 0.011883480945424218, "min": 0.011883480945424218}}, "EndTime": 1552274116.020312, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274116.020296}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011918503241323347, "sum": 0.011918503241323347, "min": 0.011918503241323347}}, "EndTime": 1552274116.020378, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274116.020361}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011878312025237921, "sum": 0.011878312025237921, "min": 0.011878312025237921}}, "EndTime": 1552274116.020435, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274116.020419}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011914209698911887, "sum": 0.011914209698911887, "min": 0.011914209698911887}}, "EndTime": 1552274116.020496, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274116.02048}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.014625513352341388, "sum": 0.014625513352341388, "min": 0.014625513352341388}}, "EndTime": 1552274116.020549, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274116.020535}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.013732083848972416, "sum": 0.013732083848972416, "min": 0.013732083848972416}}, "EndTime": 1552274116.020606, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274116.020591}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01462572515430163, "sum": 0.01462572515430163, "min": 0.01462572515430163}}, "EndTime": 1552274116.020663, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274116.020647}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.013733733051386312, "sum": 0.013733733051386312, "min": 0.013733733051386312}}, "EndTime": 1552274116.020726, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274116.02071}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012857922017873832, "sum": 0.012857922017873832, "min": 0.012857922017873832}}, "EndTime": 1552274116.020779, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274116.020764}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012884794590461194, "sum": 0.012884794590461194, "min": 0.012884794590461194}}, "EndTime": 1552274116.020836, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274116.020819}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012859059595582473, "sum": 0.012859059595582473, "min": 0.012859059595582473}}, "EndTime": 1552274116.02089, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274116.020875}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012884603715422166, "sum": 0.012884603715422166, "min": 0.012884603715422166}}, "EndTime": 1552274116.020944, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274116.020929}
    [0m
    [31m[03/11/2019 03:15:16 INFO 140227862767424] #quality_metric: host=algo-1, epoch=3, train binary_classification_cross_entropy_objective <loss>=0.00803413065714[0m
    [31m[03/11/2019 03:15:16 INFO 140227862767424] #early_stopping_criteria_metric: host=algo-1, epoch=3, criteria=binary_classification_cross_entropy_objective, value=0.00503712064417[0m
    [31m[03/11/2019 03:15:16 INFO 140227862767424] Epoch 3: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:15:16 INFO 140227862767424] #progress_metric: host=algo-1, completed 26 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 812, "sum": 812.0, "min": 812}, "Total Records Seen": {"count": 1, "max": 809456, "sum": 809456.0, "min": 809456}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 6, "sum": 6.0, "min": 6}}, "EndTime": 1552274116.023661, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274109.979635}
    [0m
    [31m[03/11/2019 03:15:16 INFO 140227862767424] #throughput_metric: host=algo-1, train throughput=32984.6768192 records/second[0m
    [31m[2019-03-11 03:15:16.023] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 5, "duration": 6044, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0069169620389315355, "sum": 0.0069169620389315355, "min": 0.0069169620389315355}}, "EndTime": 1552274122.27967, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274122.2796}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.00574524320429893, "sum": 0.00574524320429893, "min": 0.00574524320429893}}, "EndTime": 1552274122.27975, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274122.279736}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.00691596810062926, "sum": 0.00691596810062926, "min": 0.00691596810062926}}, "EndTime": 1552274122.279805, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274122.279789}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005747290889222418, "sum": 0.005747290889222418, "min": 0.005747290889222418}}, "EndTime": 1552274122.279864, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274122.279848}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0049067022006715364, "sum": 0.0049067022006715364, "min": 0.0049067022006715364}}, "EndTime": 1552274122.279902, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274122.279893}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005987407533891836, "sum": 0.005987407533891836, "min": 0.005987407533891836}}, "EndTime": 1552274122.279942, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274122.279929}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004837448609234699, "sum": 0.004837448609234699, "min": 0.004837448609234699}}, "EndTime": 1552274122.279996, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274122.279982}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005820065221295285, "sum": 0.005820065221295285, "min": 0.005820065221295285}}, "EndTime": 1552274122.280046, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274122.280033}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.007338020905777437, "sum": 0.007338020905777437, "min": 0.007338020905777437}}, "EndTime": 1552274122.280097, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274122.280082}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.006271388709245615, "sum": 0.006271388709245615, "min": 0.006271388709245615}}, "EndTime": 1552274122.28013, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274122.280122}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.007337606277897131, "sum": 0.007337606277897131, "min": 0.007337606277897131}}, "EndTime": 1552274122.280176, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274122.280165}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0062720328922846805, "sum": 0.0062720328922846805, "min": 0.0062720328922846805}}, "EndTime": 1552274122.280234, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274122.280218}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005339479718076524, "sum": 0.005339479718076524, "min": 0.005339479718076524}}, "EndTime": 1552274122.28029, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274122.280275}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005877964826984022, "sum": 0.005877964826984022, "min": 0.005877964826984022}}, "EndTime": 1552274122.280358, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274122.280343}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0053357686253648305, "sum": 0.0053357686253648305, "min": 0.0053357686253648305}}, "EndTime": 1552274122.280415, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274122.2804}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005874441825864303, "sum": 0.005874441825864303, "min": 0.005874441825864303}}, "EndTime": 1552274122.28047, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274122.280455}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01294359584309947, "sum": 0.01294359584309947, "min": 0.01294359584309947}}, "EndTime": 1552274122.280524, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274122.280509}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012251624374533419, "sum": 0.012251624374533419, "min": 0.012251624374533419}}, "EndTime": 1552274122.280577, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274122.280562}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012941720984089914, "sum": 0.012941720984089914, "min": 0.012941720984089914}}, "EndTime": 1552274122.280608, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274122.280601}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012253663547074975, "sum": 0.012253663547074975, "min": 0.012253663547074975}}, "EndTime": 1552274122.280633, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274122.280627}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011882697144345422, "sum": 0.011882697144345422, "min": 0.011882697144345422}}, "EndTime": 1552274122.280678, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274122.280664}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01192718984074329, "sum": 0.01192718984074329, "min": 0.01192718984074329}}, "EndTime": 1552274122.28073, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274122.280715}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011877108363649953, "sum": 0.011877108363649953, "min": 0.011877108363649953}}, "EndTime": 1552274122.280781, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274122.280768}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011924551403702204, "sum": 0.011924551403702204, "min": 0.011924551403702204}}, "EndTime": 1552274122.280846, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274122.280829}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.013881827797721978, "sum": 0.013881827797721978, "min": 0.013881827797721978}}, "EndTime": 1552274122.280911, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274122.280895}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01321468453311441, "sum": 0.01321468453311441, "min": 0.01321468453311441}}, "EndTime": 1552274122.280974, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274122.280958}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.013881908687514875, "sum": 0.013881908687514875, "min": 0.013881908687514875}}, "EndTime": 1552274122.281038, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274122.281022}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.013215637271727749, "sum": 0.013215637271727749, "min": 0.013215637271727749}}, "EndTime": 1552274122.281101, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274122.281085}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01285734530729265, "sum": 0.01285734530729265, "min": 0.01285734530729265}}, "EndTime": 1552274122.281158, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274122.281141}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012893029334557117, "sum": 0.012893029334557117, "min": 0.012893029334557117}}, "EndTime": 1552274122.281221, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274122.281205}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012858692809564984, "sum": 0.012858692809564984, "min": 0.012858692809564984}}, "EndTime": 1552274122.281276, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274122.281261}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012895550845855445, "sum": 0.012895550845855445, "min": 0.012895550845855445}}, "EndTime": 1552274122.281327, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274122.281312}
    [0m
    [31m[03/11/2019 03:15:22 INFO 140227862767424] #quality_metric: host=algo-1, epoch=4, train binary_classification_cross_entropy_objective <loss>=0.00691696203893[0m
    [31m[03/11/2019 03:15:22 INFO 140227862767424] #early_stopping_criteria_metric: host=algo-1, epoch=4, criteria=binary_classification_cross_entropy_objective, value=0.00483744860923[0m
    [31m[03/11/2019 03:15:22 INFO 140227862767424] Epoch 4: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:15:22 INFO 140227862767424] #progress_metric: host=algo-1, completed 33 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 1012, "sum": 1012.0, "min": 1012}, "Total Records Seen": {"count": 1, "max": 1008820, "sum": 1008820.0, "min": 1008820}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 7, "sum": 7.0, "min": 7}}, "EndTime": 1552274122.284123, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274116.023914}
    [0m
    [31m[03/11/2019 03:15:22 INFO 140227862767424] #throughput_metric: host=algo-1, train throughput=31845.6518643 records/second[0m
    [31m[2019-03-11 03:15:22.284] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 6, "duration": 6260, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.006276343655945668, "sum": 0.006276343655945668, "min": 0.006276343655945668}}, "EndTime": 1552274128.064118, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274128.064059}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005225318925464573, "sum": 0.005225318925464573, "min": 0.005225318925464573}}, "EndTime": 1552274128.064199, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274128.064181}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.006276083052457877, "sum": 0.006276083052457877, "min": 0.006276083052457877}}, "EndTime": 1552274128.064258, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274128.064242}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0052264984804182195, "sum": 0.0052264984804182195, "min": 0.0052264984804182195}}, "EndTime": 1552274128.0643, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274128.064291}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004745869061305894, "sum": 0.004745869061305894, "min": 0.004745869061305894}}, "EndTime": 1552274128.064339, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274128.064325}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.006051138314544855, "sum": 0.006051138314544855, "min": 0.006051138314544855}}, "EndTime": 1552274128.064388, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274128.064375}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004873958140761409, "sum": 0.004873958140761409, "min": 0.004873958140761409}}, "EndTime": 1552274128.06444, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274128.064425}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.006153497193401782, "sum": 0.006153497193401782, "min": 0.006153497193401782}}, "EndTime": 1552274128.064492, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274128.064478}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.006753806056688778, "sum": 0.006753806056688778, "min": 0.006753806056688778}}, "EndTime": 1552274128.064545, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274128.064533}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005827406466306754, "sum": 0.005827406466306754, "min": 0.005827406466306754}}, "EndTime": 1552274128.064585, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274128.064571}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0067534985362584865, "sum": 0.0067534985362584865, "min": 0.0067534985362584865}}, "EndTime": 1552274128.064629, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274128.064615}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005827982135154494, "sum": 0.005827982135154494, "min": 0.005827982135154494}}, "EndTime": 1552274128.064696, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274128.064678}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005348205740727372, "sum": 0.005348205740727372, "min": 0.005348205740727372}}, "EndTime": 1552274128.064757, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274128.06474}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005855620055941481, "sum": 0.005855620055941481, "min": 0.005855620055941481}}, "EndTime": 1552274128.064825, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274128.064808}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005346461805865992, "sum": 0.005346461805865992, "min": 0.005346461805865992}}, "EndTime": 1552274128.064885, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274128.064869}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005852809699336488, "sum": 0.005852809699336488, "min": 0.005852809699336488}}, "EndTime": 1552274128.064944, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274128.064928}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012566139340999738, "sum": 0.012566139340999738, "min": 0.012566139340999738}}, "EndTime": 1552274128.065012, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274128.064996}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012038233971475956, "sum": 0.012038233971475956, "min": 0.012038233971475956}}, "EndTime": 1552274128.065069, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274128.065054}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01256493431239871, "sum": 0.01256493431239871, "min": 0.01256493431239871}}, "EndTime": 1552274128.065122, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274128.065108}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012039702561632473, "sum": 0.012039702561632473, "min": 0.012039702561632473}}, "EndTime": 1552274128.065174, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274128.065159}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0118642488490397, "sum": 0.0118642488490397, "min": 0.0118642488490397}}, "EndTime": 1552274128.065221, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274128.065206}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011934922299193377, "sum": 0.011934922299193377, "min": 0.011934922299193377}}, "EndTime": 1552274128.065268, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274128.065258}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011862491466891226, "sum": 0.011862491466891226, "min": 0.011862491466891226}}, "EndTime": 1552274128.065313, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274128.0653}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011933327715001513, "sum": 0.011933327715001513, "min": 0.011933327715001513}}, "EndTime": 1552274128.06537, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274128.065354}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.013505783789121924, "sum": 0.013505783789121924, "min": 0.013505783789121924}}, "EndTime": 1552274128.065426, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274128.06541}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.013001762073842724, "sum": 0.013001762073842724, "min": 0.013001762073842724}}, "EndTime": 1552274128.065481, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274128.065466}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0135056490993979, "sum": 0.0135056490993979, "min": 0.0135056490993979}}, "EndTime": 1552274128.065545, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274128.065529}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.013002546663859381, "sum": 0.013002546663859381, "min": 0.013002546663859381}}, "EndTime": 1552274128.065591, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274128.065581}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01285640471544697, "sum": 0.01285640471544697, "min": 0.01285640471544697}}, "EndTime": 1552274128.065647, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274128.065631}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012896197460404593, "sum": 0.012896197460404593, "min": 0.012896197460404593}}, "EndTime": 1552274128.065699, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274128.065683}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012857996290652596, "sum": 0.012857996290652596, "min": 0.012857996290652596}}, "EndTime": 1552274128.065755, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274128.06574}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012898726200338584, "sum": 0.012898726200338584, "min": 0.012898726200338584}}, "EndTime": 1552274128.06581, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274128.065794}
    [0m
    [31m[03/11/2019 03:15:28 INFO 140227862767424] #quality_metric: host=algo-1, epoch=5, train binary_classification_cross_entropy_objective <loss>=0.00627634365595[0m
    [31m[03/11/2019 03:15:28 INFO 140227862767424] #early_stopping_criteria_metric: host=algo-1, epoch=5, criteria=binary_classification_cross_entropy_objective, value=0.00474586906131[0m
    [31m[03/11/2019 03:15:28 INFO 140227862767424] Epoch 5: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:15:28 INFO 140227862767424] #progress_metric: host=algo-1, completed 40 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 1212, "sum": 1212.0, "min": 1212}, "Total Records Seen": {"count": 1, "max": 1208184, "sum": 1208184.0, "min": 1208184}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 8, "sum": 8.0, "min": 8}}, "EndTime": 1552274128.068473, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274122.284393}
    [0m
    [31m[03/11/2019 03:15:28 INFO 140227862767424] #throughput_metric: host=algo-1, train throughput=34467.0136282 records/second[0m
    [31m[2019-03-11 03:15:28.068] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 7, "duration": 5784, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005869037068668922, "sum": 0.005869037068668922, "min": 0.005869037068668922}}, "EndTime": 1552274133.963651, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274133.963592}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004911643637484642, "sum": 0.004911643637484642, "min": 0.004911643637484642}}, "EndTime": 1552274133.963734, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274133.963716}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005869176341061616, "sum": 0.005869176341061616, "min": 0.005869176341061616}}, "EndTime": 1552274133.96379, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274133.963774}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004912145971053809, "sum": 0.004912145971053809, "min": 0.004912145971053809}}, "EndTime": 1552274133.963843, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274133.963828}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004701146499894972, "sum": 0.004701146499894972, "min": 0.004701146499894972}}, "EndTime": 1552274133.96389, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274133.963877}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.00605768408989487, "sum": 0.00605768408989487, "min": 0.00605768408989487}}, "EndTime": 1552274133.963937, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274133.963923}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0049241019635643795, "sum": 0.0049241019635643795, "min": 0.0049241019635643795}}, "EndTime": 1552274133.963991, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274133.963976}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005674689029628907, "sum": 0.005674689029628907, "min": 0.005674689029628907}}, "EndTime": 1552274133.964048, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274133.964032}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.006389971089722523, "sum": 0.006389971089722523, "min": 0.006389971089722523}}, "EndTime": 1552274133.964103, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274133.964087}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.00557527757050404, "sum": 0.00557527757050404, "min": 0.00557527757050404}}, "EndTime": 1552274133.964157, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274133.964142}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.00638973372306057, "sum": 0.00638973372306057, "min": 0.00638973372306057}}, "EndTime": 1552274133.964212, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274133.964197}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005575775871324779, "sum": 0.005575775871324779, "min": 0.005575775871324779}}, "EndTime": 1552274133.964267, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274133.964252}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0053742247679125724, "sum": 0.0053742247679125724, "min": 0.0053742247679125724}}, "EndTime": 1552274133.964322, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274133.964307}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005830225506020551, "sum": 0.005830225506020551, "min": 0.005830225506020551}}, "EndTime": 1552274133.964376, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274133.964361}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005373988723635074, "sum": 0.005373988723635074, "min": 0.005373988723635074}}, "EndTime": 1552274133.964438, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274133.964424}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005828681292246334, "sum": 0.005828681292246334, "min": 0.005828681292246334}}, "EndTime": 1552274133.964491, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274133.964477}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012353397164512519, "sum": 0.012353397164512519, "min": 0.012353397164512519}}, "EndTime": 1552274133.96454, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274133.964526}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011946621701945012, "sum": 0.011946621701945012, "min": 0.011946621701945012}}, "EndTime": 1552274133.964601, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274133.964585}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012352558684708486, "sum": 0.012352558684708486, "min": 0.012352558684708486}}, "EndTime": 1552274133.96466, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274133.964644}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011947648996084779, "sum": 0.011947648996084779, "min": 0.011947648996084779}}, "EndTime": 1552274133.964713, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274133.964699}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011843015904402614, "sum": 0.011843015904402614, "min": 0.011843015904402614}}, "EndTime": 1552274133.964775, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274133.964759}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011943208752565049, "sum": 0.011943208752565049, "min": 0.011943208752565049}}, "EndTime": 1552274133.964838, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274133.964823}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011843548118768626, "sum": 0.011843548118768626, "min": 0.011843548118768626}}, "EndTime": 1552274133.9649, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274133.964885}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01194197491904599, "sum": 0.01194197491904599, "min": 0.01194197491904599}}, "EndTime": 1552274133.964958, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274133.964943}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.013294168243456127, "sum": 0.013294168243456127, "min": 0.013294168243456127}}, "EndTime": 1552274133.96502, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274133.965005}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012910012778325297, "sum": 0.012910012778325297, "min": 0.012910012778325297}}, "EndTime": 1552274133.965075, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274133.965059}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.013293834397541218, "sum": 0.013293834397541218, "min": 0.013293834397541218}}, "EndTime": 1552274133.96514, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274133.965124}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012910769558432114, "sum": 0.012910769558432114, "min": 0.012910769558432114}}, "EndTime": 1552274133.965203, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274133.965187}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012855722399213206, "sum": 0.012855722399213206, "min": 0.012855722399213206}}, "EndTime": 1552274133.965267, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274133.96525}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01291352930979513, "sum": 0.01291352930979513, "min": 0.01291352930979513}}, "EndTime": 1552274133.96533, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274133.965314}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01285700117643155, "sum": 0.01285700117643155, "min": 0.01285700117643155}}, "EndTime": 1552274133.965393, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274133.965376}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01290991102031727, "sum": 0.01290991102031727, "min": 0.01290991102031727}}, "EndTime": 1552274133.965455, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274133.96544}
    [0m
    [31m[03/11/2019 03:15:33 INFO 140227862767424] #quality_metric: host=algo-1, epoch=6, train binary_classification_cross_entropy_objective <loss>=0.00586903706867[0m
    [31m[03/11/2019 03:15:33 INFO 140227862767424] #early_stopping_criteria_metric: host=algo-1, epoch=6, criteria=binary_classification_cross_entropy_objective, value=0.00470114649989[0m
    [31m[03/11/2019 03:15:33 INFO 140227862767424] Epoch 6: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:15:33 INFO 140227862767424] #progress_metric: host=algo-1, completed 46 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 1412, "sum": 1412.0, "min": 1412}, "Total Records Seen": {"count": 1, "max": 1407548, "sum": 1407548.0, "min": 1407548}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 9, "sum": 9.0, "min": 9}}, "EndTime": 1552274133.967959, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274128.068756}
    [0m
    [31m[03/11/2019 03:15:33 INFO 140227862767424] #throughput_metric: host=algo-1, train throughput=33794.3972402 records/second[0m
    [31m[2019-03-11 03:15:33.968] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 8, "duration": 5899, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005591161470317361, "sum": 0.005591161470317361, "min": 0.005591161470317361}}, "EndTime": 1552274139.960436, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274139.960377}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004708522209570036, "sum": 0.004708522209570036, "min": 0.004708522209570036}}, "EndTime": 1552274139.960511, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274139.960498}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005591602163698206, "sum": 0.005591602163698206, "min": 0.005591602163698206}}, "EndTime": 1552274139.960564, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274139.96055}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0047086963965066115, "sum": 0.0047086963965066115, "min": 0.0047086963965066115}}, "EndTime": 1552274139.960614, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274139.960598}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004754318347378592, "sum": 0.004754318347378592, "min": 0.004754318347378592}}, "EndTime": 1552274139.960655, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274139.960645}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005659493515764049, "sum": 0.005659493515764049, "min": 0.005659493515764049}}, "EndTime": 1552274139.960685, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274139.960675}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004616728931216139, "sum": 0.004616728931216139, "min": 0.004616728931216139}}, "EndTime": 1552274139.960734, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274139.96072}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.00646706811327431, "sum": 0.00646706811327431, "min": 0.00646706811327431}}, "EndTime": 1552274139.960791, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274139.960776}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.006146658008422084, "sum": 0.006146658008422084, "min": 0.006146658008422084}}, "EndTime": 1552274139.960891, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274139.960874}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005423311430006171, "sum": 0.005423311430006171, "min": 0.005423311430006171}}, "EndTime": 1552274139.960948, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274139.960933}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0061464657795489135, "sum": 0.0061464657795489135, "min": 0.0061464657795489135}}, "EndTime": 1552274139.961002, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274139.960987}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0054237357953085975, "sum": 0.0054237357953085975, "min": 0.0054237357953085975}}, "EndTime": 1552274139.961057, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274139.961042}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005315854691380831, "sum": 0.005315854691380831, "min": 0.005315854691380831}}, "EndTime": 1552274139.961113, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274139.961098}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005809691983251716, "sum": 0.005809691983251716, "min": 0.005809691983251716}}, "EndTime": 1552274139.961164, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274139.961149}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005315677183057795, "sum": 0.005315677183057795, "min": 0.005315677183057795}}, "EndTime": 1552274139.961295, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274139.961274}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005809011261367319, "sum": 0.005809011261367319, "min": 0.005809011261367319}}, "EndTime": 1552274139.961365, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274139.961348}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012224241410068531, "sum": 0.012224241410068531, "min": 0.012224241410068531}}, "EndTime": 1552274139.961427, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274139.961411}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01190716770486017, "sum": 0.01190716770486017, "min": 0.01190716770486017}}, "EndTime": 1552274139.961492, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274139.961475}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01222359538078308, "sum": 0.01222359538078308, "min": 0.01222359538078308}}, "EndTime": 1552274139.961555, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274139.961539}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011907873830603594, "sum": 0.011907873830603594, "min": 0.011907873830603594}}, "EndTime": 1552274139.961619, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274139.961602}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011822155673300201, "sum": 0.011822155673300201, "min": 0.011822155673300201}}, "EndTime": 1552274139.961677, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274139.96166}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011951955908506959, "sum": 0.011951955908506959, "min": 0.011951955908506959}}, "EndTime": 1552274139.96174, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274139.961723}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011823138146544222, "sum": 0.011823138146544222, "min": 0.011823138146544222}}, "EndTime": 1552274139.961803, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274139.961787}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011950859326214048, "sum": 0.011950859326214048, "min": 0.011950859326214048}}, "EndTime": 1552274139.961867, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274139.96185}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0131656456448924, "sum": 0.0131656456448924, "min": 0.0131656456448924}}, "EndTime": 1552274139.961927, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274139.961912}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012870260859254617, "sum": 0.012870260859254617, "min": 0.012870260859254617}}, "EndTime": 1552274139.961987, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274139.961972}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.013165357339322267, "sum": 0.013165357339322267, "min": 0.013165357339322267}}, "EndTime": 1552274139.96204, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274139.962024}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012870954639348552, "sum": 0.012870954639348552, "min": 0.012870954639348552}}, "EndTime": 1552274139.962095, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274139.96208}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012855612519398407, "sum": 0.012855612519398407, "min": 0.012855612519398407}}, "EndTime": 1552274139.962149, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274139.962133}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012923838847845643, "sum": 0.012923838847845643, "min": 0.012923838847845643}}, "EndTime": 1552274139.962203, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274139.962187}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0128559720863649, "sum": 0.0128559720863649, "min": 0.0128559720863649}}, "EndTime": 1552274139.962258, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274139.962242}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012921573549658808, "sum": 0.012921573549658808, "min": 0.012921573549658808}}, "EndTime": 1552274139.962315, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274139.9623}
    [0m
    [31m[03/11/2019 03:15:39 INFO 140227862767424] #quality_metric: host=algo-1, epoch=7, train binary_classification_cross_entropy_objective <loss>=0.00559116147032[0m
    [31m[03/11/2019 03:15:39 INFO 140227862767424] #early_stopping_criteria_metric: host=algo-1, epoch=7, criteria=binary_classification_cross_entropy_objective, value=0.00461672893122[0m
    [31m[03/11/2019 03:15:39 INFO 140227862767424] Epoch 7: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:15:39 INFO 140227862767424] #progress_metric: host=algo-1, completed 53 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 1612, "sum": 1612.0, "min": 1612}, "Total Records Seen": {"count": 1, "max": 1606912, "sum": 1606912.0, "min": 1606912}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 10, "sum": 10.0, "min": 10}}, "EndTime": 1552274139.96484, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274133.968223}
    [0m
    [31m[03/11/2019 03:15:39 INFO 140227862767424] #throughput_metric: host=algo-1, train throughput=33245.4344272 records/second[0m
    [31m[2019-03-11 03:15:39.965] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 9, "duration": 5996, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005391925841719661, "sum": 0.005391925841719661, "min": 0.005391925841719661}}, "EndTime": 1552274145.633303, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274145.633243}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0045703682456184275, "sum": 0.0045703682456184275, "min": 0.0045703682456184275}}, "EndTime": 1552274145.63338, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274145.633367}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005392487336642778, "sum": 0.005392487336642778, "min": 0.005392487336642778}}, "EndTime": 1552274145.633438, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274145.633426}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004570626125263808, "sum": 0.004570626125263808, "min": 0.004570626125263808}}, "EndTime": 1552274145.633486, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274145.633471}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004525650208619372, "sum": 0.004525650208619372, "min": 0.004525650208619372}}, "EndTime": 1552274145.633525, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274145.633516}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005934561043453576, "sum": 0.005934561043453576, "min": 0.005934561043453576}}, "EndTime": 1552274145.633552, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274145.633545}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0046104487279551715, "sum": 0.0046104487279551715, "min": 0.0046104487279551715}}, "EndTime": 1552274145.633594, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274145.633581}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0064192624762279904, "sum": 0.0064192624762279904, "min": 0.0064192624762279904}}, "EndTime": 1552274145.633646, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274145.633631}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005975344323632705, "sum": 0.005975344323632705, "min": 0.005975344323632705}}, "EndTime": 1552274145.633696, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274145.633685}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005327865115362196, "sum": 0.005327865115362196, "min": 0.005327865115362196}}, "EndTime": 1552274145.633724, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274145.633717}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.00597518141904668, "sum": 0.00597518141904668, "min": 0.00597518141904668}}, "EndTime": 1552274145.633759, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274145.633746}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005328222943909803, "sum": 0.005328222943909803, "min": 0.005328222943909803}}, "EndTime": 1552274145.63381, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274145.633797}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005267092317792039, "sum": 0.005267092317792039, "min": 0.005267092317792039}}, "EndTime": 1552274145.633864, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274145.633848}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0057899654630440564, "sum": 0.0057899654630440564, "min": 0.0057899654630440564}}, "EndTime": 1552274145.633918, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274145.633902}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005267111534748844, "sum": 0.005267111534748844, "min": 0.005267111534748844}}, "EndTime": 1552274145.633971, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274145.633956}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005789906225611816, "sum": 0.005789906225611816, "min": 0.005789906225611816}}, "EndTime": 1552274145.634035, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274145.63402}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012141525111605773, "sum": 0.012141525111605773, "min": 0.012141525111605773}}, "EndTime": 1552274145.634084, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274145.634073}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011890671925928126, "sum": 0.011890671925928126, "min": 0.011890671925928126}}, "EndTime": 1552274145.634112, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274145.634105}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012141005372282248, "sum": 0.012141005372282248, "min": 0.012141005372282248}}, "EndTime": 1552274145.634156, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274145.634145}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011891155160252173, "sum": 0.011891155160252173, "min": 0.011891155160252173}}, "EndTime": 1552274145.634204, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274145.634191}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011817079468588135, "sum": 0.011817079468588135, "min": 0.011817079468588135}}, "EndTime": 1552274145.634254, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274145.634239}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011961612669666808, "sum": 0.011961612669666808, "min": 0.011961612669666808}}, "EndTime": 1552274145.634307, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274145.634292}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011818041877530928, "sum": 0.011818041877530928, "min": 0.011818041877530928}}, "EndTime": 1552274145.634359, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274145.634345}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011960513301231154, "sum": 0.011960513301231154, "min": 0.011960513301231154}}, "EndTime": 1552274145.634411, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274145.634397}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01308310868153021, "sum": 0.01308310868153021, "min": 0.01308310868153021}}, "EndTime": 1552274145.634463, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274145.634448}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012853279242563487, "sum": 0.012853279242563487, "min": 0.012853279242563487}}, "EndTime": 1552274145.634517, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274145.634503}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.013083011689497598, "sum": 0.013083011689497598, "min": 0.013083011689497598}}, "EndTime": 1552274145.634571, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274145.634556}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012853929442376946, "sum": 0.012853929442376946, "min": 0.012853929442376946}}, "EndTime": 1552274145.634617, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274145.634603}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01285552091574549, "sum": 0.01285552091574549, "min": 0.01285552091574549}}, "EndTime": 1552274145.63467, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274145.634655}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012931668805117582, "sum": 0.012931668805117582, "min": 0.012931668805117582}}, "EndTime": 1552274145.634732, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274145.634716}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012855314294896534, "sum": 0.012855314294896534, "min": 0.012855314294896534}}, "EndTime": 1552274145.634789, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274145.634773}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012921399402258983, "sum": 0.012921399402258983, "min": 0.012921399402258983}}, "EndTime": 1552274145.634854, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274145.634837}
    [0m
    [31m[03/11/2019 03:15:45 INFO 140227862767424] #quality_metric: host=algo-1, epoch=8, train binary_classification_cross_entropy_objective <loss>=0.00539192584172[0m
    [31m[03/11/2019 03:15:45 INFO 140227862767424] #early_stopping_criteria_metric: host=algo-1, epoch=8, criteria=binary_classification_cross_entropy_objective, value=0.00452565020862[0m
    [31m[03/11/2019 03:15:45 INFO 140227862767424] Epoch 8: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:15:45 INFO 140227862767424] #progress_metric: host=algo-1, completed 60 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 1812, "sum": 1812.0, "min": 1812}, "Total Records Seen": {"count": 1, "max": 1806276, "sum": 1806276.0, "min": 1806276}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 11, "sum": 11.0, "min": 11}}, "EndTime": 1552274145.637459, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274139.965115}
    [0m
    [31m[03/11/2019 03:15:45 INFO 140227862767424] #throughput_metric: host=algo-1, train throughput=35146.0079076 records/second[0m
    [31m[2019-03-11 03:15:45.637] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 10, "duration": 5672, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005243560808387833, "sum": 0.005243560808387833, "min": 0.005243560808387833}}, "EndTime": 1552274151.846002, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274151.845943}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004472267618430919, "sum": 0.004472267618430919, "min": 0.004472267618430919}}, "EndTime": 1552274151.846084, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274151.846066}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005244257314121304, "sum": 0.005244257314121304, "min": 0.005244257314121304}}, "EndTime": 1552274151.846145, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274151.846128}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004471835252927176, "sum": 0.004471835252927176, "min": 0.004471835252927176}}, "EndTime": 1552274151.846188, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274151.846178}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004443225746628028, "sum": 0.004443225746628028, "min": 0.004443225746628028}}, "EndTime": 1552274151.846235, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274151.84622}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0056344015265829, "sum": 0.0056344015265829, "min": 0.0056344015265829}}, "EndTime": 1552274151.846289, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274151.846274}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004439634162876474, "sum": 0.004439634162876474, "min": 0.004439634162876474}}, "EndTime": 1552274151.846345, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274151.846329}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005886269171857954, "sum": 0.005886269171857954, "min": 0.005886269171857954}}, "EndTime": 1552274151.846401, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274151.846385}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005849934860689556, "sum": 0.005849934860689556, "min": 0.005849934860689556}}, "EndTime": 1552274151.846455, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274151.84644}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005266024878875694, "sum": 0.005266024878875694, "min": 0.005266024878875694}}, "EndTime": 1552274151.846507, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274151.846492}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005849792432545417, "sum": 0.005849792432545417, "min": 0.005849792432545417}}, "EndTime": 1552274151.846574, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274151.846557}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0052663132647174085, "sum": 0.0052663132647174085, "min": 0.0052663132647174085}}, "EndTime": 1552274151.846628, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274151.846612}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005242697265279952, "sum": 0.005242697265279952, "min": 0.005242697265279952}}, "EndTime": 1552274151.846683, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274151.846667}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.00577369510797999, "sum": 0.00577369510797999, "min": 0.00577369510797999}}, "EndTime": 1552274151.846737, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274151.846722}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005243133715648747, "sum": 0.005243133715648747, "min": 0.005243133715648747}}, "EndTime": 1552274151.846794, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274151.846777}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005774005268985902, "sum": 0.005774005268985902, "min": 0.005774005268985902}}, "EndTime": 1552274151.846849, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274151.846833}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012086536772886113, "sum": 0.012086536772886113, "min": 0.012086536772886113}}, "EndTime": 1552274151.846916, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274151.8469}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011884216188186377, "sum": 0.011884216188186377, "min": 0.011884216188186377}}, "EndTime": 1552274151.846967, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274151.846952}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012086114896601769, "sum": 0.012086114896601769, "min": 0.012086114896601769}}, "EndTime": 1552274151.847022, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274151.847006}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011884549309859924, "sum": 0.011884549309859924, "min": 0.011884549309859924}}, "EndTime": 1552274151.847078, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274151.847062}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011817668989675128, "sum": 0.011817668989675128, "min": 0.011817668989675128}}, "EndTime": 1552274151.847125, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274151.84711}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011972069397044541, "sum": 0.011972069397044541, "min": 0.011972069397044541}}, "EndTime": 1552274151.84718, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274151.847165}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011817923034255828, "sum": 0.011817923034255828, "min": 0.011817923034255828}}, "EndTime": 1552274151.847242, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274151.847226}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011971008896827697, "sum": 0.011971008896827697, "min": 0.011971008896827697}}, "EndTime": 1552274151.8473, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274151.847285}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.013027740849921452, "sum": 0.013027740849921452, "min": 0.013027740849921452}}, "EndTime": 1552274151.847364, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274151.847347}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012846210579776285, "sum": 0.012846210579776285, "min": 0.012846210579776285}}, "EndTime": 1552274151.847449, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274151.847431}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.013027749891856209, "sum": 0.013027749891856209, "min": 0.013027749891856209}}, "EndTime": 1552274151.847501, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274151.847488}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0128467862899579, "sum": 0.0128467862899579, "min": 0.0128467862899579}}, "EndTime": 1552274151.847553, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274151.84754}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012855197807053225, "sum": 0.012855197807053225, "min": 0.012855197807053225}}, "EndTime": 1552274151.847587, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274151.847578}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012939482097649694, "sum": 0.012939482097649694, "min": 0.012939482097649694}}, "EndTime": 1552274151.847615, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274151.847608}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012855091443612948, "sum": 0.012855091443612948, "min": 0.012855091443612948}}, "EndTime": 1552274151.847651, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274151.847638}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012936638314520294, "sum": 0.012936638314520294, "min": 0.012936638314520294}}, "EndTime": 1552274151.847707, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274151.847692}
    [0m
    [31m[03/11/2019 03:15:51 INFO 140227862767424] #quality_metric: host=algo-1, epoch=9, train binary_classification_cross_entropy_objective <loss>=0.00524356080839[0m
    [31m[03/11/2019 03:15:51 INFO 140227862767424] #early_stopping_criteria_metric: host=algo-1, epoch=9, criteria=binary_classification_cross_entropy_objective, value=0.00443963416288[0m
    [31m[03/11/2019 03:15:51 INFO 140227862767424] Epoch 9: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:15:51 INFO 140227862767424] #progress_metric: host=algo-1, completed 66 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 2012, "sum": 2012.0, "min": 2012}, "Total Records Seen": {"count": 1, "max": 2005640, "sum": 2005640.0, "min": 2005640}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 12, "sum": 12.0, "min": 12}}, "EndTime": 1552274151.850271, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274145.637728}
    [0m
    [31m[03/11/2019 03:15:51 INFO 140227862767424] #throughput_metric: host=algo-1, train throughput=32089.9172239 records/second[0m
    [31m[2019-03-11 03:15:51.850] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 11, "duration": 6212, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005129939455482828, "sum": 0.005129939455482828, "min": 0.005129939455482828}}, "EndTime": 1552274157.853806, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274157.853748}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004400380541631325, "sum": 0.004400380541631325, "min": 0.004400380541631325}}, "EndTime": 1552274157.853879, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274157.853866}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005130695801284445, "sum": 0.005130695801284445, "min": 0.005130695801284445}}, "EndTime": 1552274157.853937, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274157.85392}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004400535478364283, "sum": 0.004400535478364283, "min": 0.004400535478364283}}, "EndTime": 1552274157.853984, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274157.85397}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0044496707453500085, "sum": 0.0044496707453500085, "min": 0.0044496707453500085}}, "EndTime": 1552274157.854034, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274157.854023}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.006058443752964537, "sum": 0.006058443752964537, "min": 0.006058443752964537}}, "EndTime": 1552274157.854074, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274157.854059}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004425278927513103, "sum": 0.004425278927513103, "min": 0.004425278927513103}}, "EndTime": 1552274157.85414, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274157.854122}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.00605832583618224, "sum": 0.00605832583618224, "min": 0.00605832583618224}}, "EndTime": 1552274157.854208, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274157.85419}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005755311613106847, "sum": 0.005755311613106847, "min": 0.005755311613106847}}, "EndTime": 1552274157.854274, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274157.854257}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005224952253864039, "sum": 0.005224952253864039, "min": 0.005224952253864039}}, "EndTime": 1552274157.854339, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274157.854322}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005755185417793504, "sum": 0.005755185417793504, "min": 0.005755185417793504}}, "EndTime": 1552274157.854403, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274157.854387}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0052251979810508655, "sum": 0.0052251979810508655, "min": 0.0052251979810508655}}, "EndTime": 1552274157.854468, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274157.854452}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005231819331645965, "sum": 0.005231819331645965, "min": 0.005231819331645965}}, "EndTime": 1552274157.854544, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274157.854527}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005756311737412784, "sum": 0.005756311737412784, "min": 0.005756311737412784}}, "EndTime": 1552274157.854596, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274157.854582}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005232467777465456, "sum": 0.005232467777465456, "min": 0.005232467777465456}}, "EndTime": 1552274157.854654, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274157.854638}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005756838018870234, "sum": 0.005756838018870234, "min": 0.005756838018870234}}, "EndTime": 1552274157.85471, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274157.854695}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01204894584387391, "sum": 0.01204894584387391, "min": 0.01204894584387391}}, "EndTime": 1552274157.854771, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274157.854755}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011882044413580966, "sum": 0.011882044413580966, "min": 0.011882044413580966}}, "EndTime": 1552274157.854828, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274157.854813}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012048594359776482, "sum": 0.012048594359776482, "min": 0.012048594359776482}}, "EndTime": 1552274157.854876, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274157.854861}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011882273570377024, "sum": 0.011882273570377024, "min": 0.011882273570377024}}, "EndTime": 1552274157.854928, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274157.854914}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011827303311932626, "sum": 0.011827303311932626, "min": 0.011827303311932626}}, "EndTime": 1552274157.854981, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274157.854966}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011982332607010501, "sum": 0.011982332607010501, "min": 0.011982332607010501}}, "EndTime": 1552274157.855035, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274157.85502}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011827303381421458, "sum": 0.011827303381421458, "min": 0.011827303381421458}}, "EndTime": 1552274157.855089, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274157.855074}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011981394795317147, "sum": 0.011981394795317147, "min": 0.011981394795317147}}, "EndTime": 1552274157.855147, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274157.855131}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012989261237820189, "sum": 0.012989261237820189, "min": 0.012989261237820189}}, "EndTime": 1552274157.855208, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274157.855192}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012843442471782167, "sum": 0.012843442471782167, "min": 0.012843442471782167}}, "EndTime": 1552274157.855265, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274157.85525}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012989297969856454, "sum": 0.012989297969856454, "min": 0.012989297969856454}}, "EndTime": 1552274157.855324, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274157.855312}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012843923484260713, "sum": 0.012843923484260713, "min": 0.012843923484260713}}, "EndTime": 1552274157.855377, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274157.855362}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012854552010195938, "sum": 0.012854552010195938, "min": 0.012854552010195938}}, "EndTime": 1552274157.855452, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274157.855435}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012953675210775442, "sum": 0.012953675210775442, "min": 0.012953675210775442}}, "EndTime": 1552274157.85551, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274157.855494}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01285420082262413, "sum": 0.01285420082262413, "min": 0.01285420082262413}}, "EndTime": 1552274157.855562, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274157.855547}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012952155484626042, "sum": 0.012952155484626042, "min": 0.012952155484626042}}, "EndTime": 1552274157.855618, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274157.855602}
    [0m
    [31m[03/11/2019 03:15:57 INFO 140227862767424] #quality_metric: host=algo-1, epoch=10, train binary_classification_cross_entropy_objective <loss>=0.00512993945548[0m
    [31m[03/11/2019 03:15:57 INFO 140227862767424] #early_stopping_criteria_metric: host=algo-1, epoch=10, criteria=binary_classification_cross_entropy_objective, value=0.00440038054163[0m
    [31m[03/11/2019 03:15:57 INFO 140227862767424] Epoch 10: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:15:57 INFO 140227862767424] #progress_metric: host=algo-1, completed 73 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 2212, "sum": 2212.0, "min": 2212}, "Total Records Seen": {"count": 1, "max": 2205004, "sum": 2205004.0, "min": 2205004}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 13, "sum": 13.0, "min": 13}}, "EndTime": 1552274157.858214, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274151.850533}
    [0m
    [31m[03/11/2019 03:15:57 INFO 140227862767424] #throughput_metric: host=algo-1, train throughput=33184.2380916 records/second[0m
    [31m[2019-03-11 03:15:57.858] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 12, "duration": 6007, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0050408838328404645, "sum": 0.0050408838328404645, "min": 0.0050408838328404645}}, "EndTime": 1552274163.773514, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274163.773458}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004347342520802464, "sum": 0.004347342520802464, "min": 0.004347342520802464}}, "EndTime": 1552274163.773588, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274163.773575}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005041672517905882, "sum": 0.005041672517905882, "min": 0.005041672517905882}}, "EndTime": 1552274163.773642, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274163.773627}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004347448597601311, "sum": 0.004347448597601311, "min": 0.004347448597601311}}, "EndTime": 1552274163.773699, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274163.773682}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004311184193021688, "sum": 0.004311184193021688, "min": 0.004311184193021688}}, "EndTime": 1552274163.773752, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274163.773737}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0057367004239229705, "sum": 0.0057367004239229705, "min": 0.0057367004239229705}}, "EndTime": 1552274163.773807, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274163.773792}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004303851319617362, "sum": 0.004303851319617362, "min": 0.004303851319617362}}, "EndTime": 1552274163.773861, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274163.773846}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005506682152349745, "sum": 0.005506682152349745, "min": 0.005506682152349745}}, "EndTime": 1552274163.773912, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274163.773897}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005682184379304474, "sum": 0.005682184379304474, "min": 0.005682184379304474}}, "EndTime": 1552274163.773965, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274163.77395}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005197159426895219, "sum": 0.005197159426895219, "min": 0.005197159426895219}}, "EndTime": 1552274163.77402, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274163.774004}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005682071232316482, "sum": 0.005682071232316482, "min": 0.005682071232316482}}, "EndTime": 1552274163.774075, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274163.774059}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005197354152274491, "sum": 0.005197354152274491, "min": 0.005197354152274491}}, "EndTime": 1552274163.77413, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274163.774115}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0052181945592913795, "sum": 0.0052181945592913795, "min": 0.0052181945592913795}}, "EndTime": 1552274163.774186, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274163.77417}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005725392285902895, "sum": 0.005725392285902895, "min": 0.005725392285902895}}, "EndTime": 1552274163.774255, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274163.774239}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005218749440794614, "sum": 0.005218749440794614, "min": 0.005218749440794614}}, "EndTime": 1552274163.774309, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274163.774293}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005725960174397608, "sum": 0.005725960174397608, "min": 0.005725960174397608}}, "EndTime": 1552274163.774363, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274163.774347}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012022629658780506, "sum": 0.012022629658780506, "min": 0.012022629658780506}}, "EndTime": 1552274163.774425, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274163.774409}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011881658866177851, "sum": 0.011881658866177851, "min": 0.011881658866177851}}, "EndTime": 1552274163.77448, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274163.774465}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012022314350808685, "sum": 0.012022314350808685, "min": 0.012022314350808685}}, "EndTime": 1552274163.774533, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274163.774518}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011881814344444467, "sum": 0.011881814344444467, "min": 0.011881814344444467}}, "EndTime": 1552274163.774586, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274163.774571}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011839835171723486, "sum": 0.011839835171723486, "min": 0.011839835171723486}}, "EndTime": 1552274163.77465, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274163.774634}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011991599743090683, "sum": 0.011991599743090683, "min": 0.011991599743090683}}, "EndTime": 1552274163.774704, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274163.774689}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011839938913757477, "sum": 0.011839938913757477, "min": 0.011839938913757477}}, "EndTime": 1552274163.774758, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274163.774743}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011990834723165885, "sum": 0.011990834723165885, "min": 0.011990834723165885}}, "EndTime": 1552274163.774811, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274163.774796}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012961713818449471, "sum": 0.012961713818449471, "min": 0.012961713818449471}}, "EndTime": 1552274163.774875, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274163.774858}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01284264255169049, "sum": 0.01284264255169049, "min": 0.01284264255169049}}, "EndTime": 1552274163.774938, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274163.774922}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012961710998161353, "sum": 0.012961710998161353, "min": 0.012961710998161353}}, "EndTime": 1552274163.775002, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274163.774985}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012842846664352033, "sum": 0.012842846664352033, "min": 0.012842846664352033}}, "EndTime": 1552274163.775059, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274163.775043}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012853404417109849, "sum": 0.012853404417109849, "min": 0.012853404417109849}}, "EndTime": 1552274163.775112, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274163.775096}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012962785278732453, "sum": 0.012962785278732453, "min": 0.012962785278732453}}, "EndTime": 1552274163.775174, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274163.775158}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012852913996682095, "sum": 0.012852913996682095, "min": 0.012852913996682095}}, "EndTime": 1552274163.775227, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274163.775212}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012966532389722278, "sum": 0.012966532389722278, "min": 0.012966532389722278}}, "EndTime": 1552274163.775279, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274163.775269}
    [0m
    [31m[03/11/2019 03:16:03 INFO 140227862767424] #quality_metric: host=algo-1, epoch=11, train binary_classification_cross_entropy_objective <loss>=0.00504088383284[0m
    [31m[03/11/2019 03:16:03 INFO 140227862767424] #early_stopping_criteria_metric: host=algo-1, epoch=11, criteria=binary_classification_cross_entropy_objective, value=0.00430385131962[0m
    [31m[03/11/2019 03:16:03 INFO 140227862767424] Epoch 11: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:16:03 INFO 140227862767424] #progress_metric: host=algo-1, completed 80 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 2412, "sum": 2412.0, "min": 2412}, "Total Records Seen": {"count": 1, "max": 2404368, "sum": 2404368.0, "min": 2404368}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 14, "sum": 14.0, "min": 14}}, "EndTime": 1552274163.777765, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274157.858477}
    [0m
    [31m[03/11/2019 03:16:03 INFO 140227862767424] #throughput_metric: host=algo-1, train throughput=33679.6795119 records/second[0m
    [31m[2019-03-11 03:16:03.777] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 13, "duration": 5919, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004969843264201179, "sum": 0.004969843264201179, "min": 0.004969843264201179}}, "EndTime": 1552274170.000911, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274170.000854}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004306342939935138, "sum": 0.004306342939935138, "min": 0.004306342939935138}}, "EndTime": 1552274170.000993, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274170.000975}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004970650205660106, "sum": 0.004970650205660106, "min": 0.004970650205660106}}, "EndTime": 1552274170.001062, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274170.001045}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004306516928888445, "sum": 0.004306516928888445, "min": 0.004306516928888445}}, "EndTime": 1552274170.001126, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274170.001109}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.00428055602671513, "sum": 0.00428055602671513, "min": 0.00428055602671513}}, "EndTime": 1552274170.001194, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274170.001176}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.006004417111451302, "sum": 0.006004417111451302, "min": 0.006004417111451302}}, "EndTime": 1552274170.00126, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274170.001243}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004273409339501031, "sum": 0.004273409339501031, "min": 0.004273409339501031}}, "EndTime": 1552274170.001326, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274170.001309}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005807543110158575, "sum": 0.005807543110158575, "min": 0.005807543110158575}}, "EndTime": 1552274170.001379, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274170.001364}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005624567857938795, "sum": 0.005624567857938795, "min": 0.005624567857938795}}, "EndTime": 1552274170.001428, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274170.001414}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005178005310758274, "sum": 0.005178005310758274, "min": 0.005178005310758274}}, "EndTime": 1552274170.00149, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274170.001473}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005624464228524635, "sum": 0.005624464228524635, "min": 0.005624464228524635}}, "EndTime": 1552274170.001549, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274170.001532}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005178166065683317, "sum": 0.005178166065683317, "min": 0.005178166065683317}}, "EndTime": 1552274170.001608, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274170.001592}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005200647512272974, "sum": 0.005200647512272974, "min": 0.005200647512272974}}, "EndTime": 1552274170.001664, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274170.001649}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0057072182270749726, "sum": 0.0057072182270749726, "min": 0.0057072182270749726}}, "EndTime": 1552274170.001727, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274170.001712}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005200967681767353, "sum": 0.005200967681767353, "min": 0.005200967681767353}}, "EndTime": 1552274170.001773, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274170.001764}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005707736464002025, "sum": 0.005707736464002025, "min": 0.005707736464002025}}, "EndTime": 1552274170.001807, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274170.001794}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012003761139347325, "sum": 0.012003761139347325, "min": 0.012003761139347325}}, "EndTime": 1552274170.001847, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274170.001838}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011881993036174295, "sum": 0.011881993036174295, "min": 0.011881993036174295}}, "EndTime": 1552274170.001873, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274170.001866}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012003464668839421, "sum": 0.012003464668839421, "min": 0.012003464668839421}}, "EndTime": 1552274170.001897, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274170.001891}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011882093836913755, "sum": 0.011882093836913755, "min": 0.011882093836913755}}, "EndTime": 1552274170.001922, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274170.001915}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011850788091295328, "sum": 0.011850788091295328, "min": 0.011850788091295328}}, "EndTime": 1552274170.001971, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274170.001957}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011999677057841315, "sum": 0.011999677057841315, "min": 0.011999677057841315}}, "EndTime": 1552274170.002001, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274170.001994}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011850821938946019, "sum": 0.011850821938946019, "min": 0.011850821938946019}}, "EndTime": 1552274170.002049, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274170.002034}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011999078996217431, "sum": 0.011999078996217431, "min": 0.011999078996217431}}, "EndTime": 1552274170.002105, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274170.002089}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012941368530743087, "sum": 0.012941368530743087, "min": 0.012941368530743087}}, "EndTime": 1552274170.00216, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274170.002145}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0128426011369456, "sum": 0.0128426011369456, "min": 0.0128426011369456}}, "EndTime": 1552274170.002211, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274170.002197}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012941372148954688, "sum": 0.012941372148954688, "min": 0.012941372148954688}}, "EndTime": 1552274170.002262, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274170.002248}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012842444234157927, "sum": 0.012842444234157927, "min": 0.012842444234157927}}, "EndTime": 1552274170.002317, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274170.002302}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012852127319604308, "sum": 0.012852127319604308, "min": 0.012852127319604308}}, "EndTime": 1552274170.002372, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274170.002357}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012978015047221927, "sum": 0.012978015047221927, "min": 0.012978015047221927}}, "EndTime": 1552274170.002436, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274170.00242}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012851788702921653, "sum": 0.012851788702921653, "min": 0.012851788702921653}}, "EndTime": 1552274170.002492, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274170.002477}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012976963284626677, "sum": 0.012976963284626677, "min": 0.012976963284626677}}, "EndTime": 1552274170.002555, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274170.002539}
    [0m
    [31m[03/11/2019 03:16:10 INFO 140227862767424] #quality_metric: host=algo-1, epoch=12, train binary_classification_cross_entropy_objective <loss>=0.0049698432642[0m
    [31m[03/11/2019 03:16:10 INFO 140227862767424] #early_stopping_criteria_metric: host=algo-1, epoch=12, criteria=binary_classification_cross_entropy_objective, value=0.0042734093395[0m
    [31m[03/11/2019 03:16:10 INFO 140227862767424] Epoch 12: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:16:10 INFO 140227862767424] #progress_metric: host=algo-1, completed 86 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 2612, "sum": 2612.0, "min": 2612}, "Total Records Seen": {"count": 1, "max": 2603732, "sum": 2603732.0, "min": 2603732}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 15, "sum": 15.0, "min": 15}}, "EndTime": 1552274170.005009, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274163.778055}
    [0m
    [31m[03/11/2019 03:16:10 INFO 140227862767424] #throughput_metric: host=algo-1, train throughput=32015.6276221 records/second[0m
    [31m[2019-03-11 03:16:10.005] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 14, "duration": 6227, "num_examples": 200}[0m
    
    2019-03-11 03:16:26 Uploading - Uploading generated training model[31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004912319385825689, "sum": 0.004912319385825689, "min": 0.004912319385825689}}, "EndTime": 1552274176.118563, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274176.118504}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004274392560199278, "sum": 0.004274392560199278, "min": 0.004274392560199278}}, "EndTime": 1552274176.118635, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274176.118623}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004913153540548967, "sum": 0.004913153540548967, "min": 0.004913153540548967}}, "EndTime": 1552274176.118689, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274176.118673}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.00427452319710698, "sum": 0.00427452319710698, "min": 0.00427452319710698}}, "EndTime": 1552274176.118745, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274176.11873}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004213615064195652, "sum": 0.004213615064195652, "min": 0.004213615064195652}}, "EndTime": 1552274176.118799, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274176.118783}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.006460462951480444, "sum": 0.006460462951480444, "min": 0.006460462951480444}}, "EndTime": 1552274176.118855, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274176.118838}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004210366838840983, "sum": 0.004210366838840983, "min": 0.004210366838840983}}, "EndTime": 1552274176.118911, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274176.118895}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005640471553607801, "sum": 0.005640471553607801, "min": 0.005640471553607801}}, "EndTime": 1552274176.118978, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274176.11896}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0055784549036217695, "sum": 0.0055784549036217695, "min": 0.0055784549036217695}}, "EndTime": 1552274176.119037, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274176.119021}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0051646082239534385, "sum": 0.0051646082239534385, "min": 0.0051646082239534385}}, "EndTime": 1552274176.119098, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274176.119082}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005578360546174361, "sum": 0.005578360546174361, "min": 0.005578360546174361}}, "EndTime": 1552274176.119157, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274176.119141}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0051647417164927155, "sum": 0.0051647417164927155, "min": 0.0051647417164927155}}, "EndTime": 1552274176.119216, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274176.1192}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0051832432740896795, "sum": 0.0051832432740896795, "min": 0.0051832432740896795}}, "EndTime": 1552274176.119284, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274176.119258}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005687902828556808, "sum": 0.005687902828556808, "min": 0.005687902828556808}}, "EndTime": 1552274176.11934, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274176.119325}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005183386637337843, "sum": 0.005183386637337843, "min": 0.005183386637337843}}, "EndTime": 1552274176.119397, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274176.119382}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005688361336537941, "sum": 0.005688361336537941, "min": 0.005688361336537941}}, "EndTime": 1552274176.119486, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274176.119469}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011989877118537174, "sum": 0.011989877118537174, "min": 0.011989877118537174}}, "EndTime": 1552274176.11955, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274176.119533}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011882599697640195, "sum": 0.011882599697640195, "min": 0.011882599697640195}}, "EndTime": 1552274176.119615, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274176.119598}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011989592566562058, "sum": 0.011989592566562058, "min": 0.011989592566562058}}, "EndTime": 1552274176.119682, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274176.119665}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01188265965691763, "sum": 0.01188265965691763, "min": 0.01188265965691763}}, "EndTime": 1552274176.119741, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274176.119724}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011865591880065113, "sum": 0.011865591880065113, "min": 0.011865591880065113}}, "EndTime": 1552274176.119796, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274176.11978}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012006625869765353, "sum": 0.012006625869765353, "min": 0.012006625869765353}}, "EndTime": 1552274176.11985, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274176.119835}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011865580839727392, "sum": 0.011865580839727392, "min": 0.011865580839727392}}, "EndTime": 1552274176.119914, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274176.119898}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012006160519230905, "sum": 0.012006160519230905, "min": 0.012006160519230905}}, "EndTime": 1552274176.119978, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274176.119962}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012926021113467577, "sum": 0.012926021113467577, "min": 0.012926021113467577}}, "EndTime": 1552274176.120041, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274176.120025}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012842781799522477, "sum": 0.012842781799522477, "min": 0.012842781799522477}}, "EndTime": 1552274176.120105, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274176.120089}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01292604173008521, "sum": 0.01292604173008521, "min": 0.01292604173008521}}, "EndTime": 1552274176.120162, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274176.120148}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012842306256294251, "sum": 0.012842306256294251, "min": 0.012842306256294251}}, "EndTime": 1552274176.120226, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274176.120209}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012850760931345686, "sum": 0.012850760931345686, "min": 0.012850760931345686}}, "EndTime": 1552274176.120291, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274176.120275}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01298285104581459, "sum": 0.01298285104581459, "min": 0.01298285104581459}}, "EndTime": 1552274176.120355, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274176.120339}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012851034403446331, "sum": 0.012851034403446331, "min": 0.012851034403446331}}, "EndTime": 1552274176.1204, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274176.12039}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01298556806693724, "sum": 0.01298556806693724, "min": 0.01298556806693724}}, "EndTime": 1552274176.120445, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274176.120431}
    [0m
    [31m[03/11/2019 03:16:16 INFO 140227862767424] #quality_metric: host=algo-1, epoch=13, train binary_classification_cross_entropy_objective <loss>=0.00491231938583[0m
    [31m[03/11/2019 03:16:16 INFO 140227862767424] #early_stopping_criteria_metric: host=algo-1, epoch=13, criteria=binary_classification_cross_entropy_objective, value=0.00421036683884[0m
    [31m[03/11/2019 03:16:16 INFO 140227862767424] Epoch 13: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:16:16 INFO 140227862767424] #progress_metric: host=algo-1, completed 93 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 2812, "sum": 2812.0, "min": 2812}, "Total Records Seen": {"count": 1, "max": 2803096, "sum": 2803096.0, "min": 2803096}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 16, "sum": 16.0, "min": 16}}, "EndTime": 1552274176.1229, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274170.005287}
    [0m
    [31m[03/11/2019 03:16:16 INFO 140227862767424] #throughput_metric: host=algo-1, train throughput=32587.8299274 records/second[0m
    [31m[2019-03-11 03:16:16.123] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 15, "duration": 6117, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0048651198454238665, "sum": 0.0048651198454238665, "min": 0.0048651198454238665}}, "EndTime": 1552274182.30443, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274182.304374}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004249175175650036, "sum": 0.004249175175650036, "min": 0.004249175175650036}}, "EndTime": 1552274182.30451, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274182.304491}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004865984468004811, "sum": 0.004865984468004811, "min": 0.004865984468004811}}, "EndTime": 1552274182.304575, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274182.304558}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004249360133056066, "sum": 0.004249360133056066, "min": 0.004249360133056066}}, "EndTime": 1552274182.304622, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274182.304612}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004188439261673683, "sum": 0.004188439261673683, "min": 0.004188439261673683}}, "EndTime": 1552274182.304652, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274182.304644}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005883642438293701, "sum": 0.005883642438293701, "min": 0.005883642438293701}}, "EndTime": 1552274182.304678, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274182.304671}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.004183407771078187, "sum": 0.004183407771078187, "min": 0.004183407771078187}}, "EndTime": 1552274182.304704, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274182.304697}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.006083663204656774, "sum": 0.006083663204656774, "min": 0.006083663204656774}}, "EndTime": 1552274182.30473, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274182.304722}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005541083724654499, "sum": 0.005541083724654499, "min": 0.005541083724654499}}, "EndTime": 1552274182.30478, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274182.304768}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005155116175886374, "sum": 0.005155116175886374, "min": 0.005155116175886374}}, "EndTime": 1552274182.304814, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274182.304803}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.00554099817491656, "sum": 0.00554099817491656, "min": 0.00554099817491656}}, "EndTime": 1552274182.304858, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274182.304848}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005155223439386742, "sum": 0.005155223439386742, "min": 0.005155223439386742}}, "EndTime": 1552274182.304884, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274182.304878}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005167508804318893, "sum": 0.005167508804318893, "min": 0.005167508804318893}}, "EndTime": 1552274182.30491, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274182.304903}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005668988515384233, "sum": 0.005668988515384233, "min": 0.005668988515384233}}, "EndTime": 1552274182.304934, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274182.304928}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005167576271385404, "sum": 0.005167576271385404, "min": 0.005167576271385404}}, "EndTime": 1552274182.304959, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274182.304952}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.005669342463040471, "sum": 0.005669342463040471, "min": 0.005669342463040471}}, "EndTime": 1552274182.304983, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274182.304977}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011979322374765598, "sum": 0.011979322374765598, "min": 0.011979322374765598}}, "EndTime": 1552274182.305008, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274182.305001}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011883317816197572, "sum": 0.011883317816197572, "min": 0.011883317816197572}}, "EndTime": 1552274182.305032, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274182.305026}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011979054548033517, "sum": 0.011979054548033517, "min": 0.011979054548033517}}, "EndTime": 1552274182.305057, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274182.305051}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011883347391483173, "sum": 0.011883347391483173, "min": 0.011883347391483173}}, "EndTime": 1552274182.305082, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274182.305075}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011879505603157696, "sum": 0.011879505603157696, "min": 0.011879505603157696}}, "EndTime": 1552274182.305134, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274182.305118}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012012594955650406, "sum": 0.012012594955650406, "min": 0.012012594955650406}}, "EndTime": 1552274182.305193, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274182.305177}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.011879369377490864, "sum": 0.011879369377490864, "min": 0.011879369377490864}}, "EndTime": 1552274182.305258, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274182.305243}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012012267659057925, "sum": 0.012012267659057925, "min": 0.012012267659057925}}, "EndTime": 1552274182.305309, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274182.305295}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012914269195729165, "sum": 0.012914269195729165, "min": 0.012914269195729165}}, "EndTime": 1552274182.305363, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274182.305348}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012842971407588402, "sum": 0.012842971407588402, "min": 0.012842971407588402}}, "EndTime": 1552274182.305419, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274182.305404}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012914280618255463, "sum": 0.012914280618255463, "min": 0.012914280618255463}}, "EndTime": 1552274182.305476, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274182.305461}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012842341140886048, "sum": 0.012842341140886048, "min": 0.012842341140886048}}, "EndTime": 1552274182.305531, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274182.305516}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.01284956756249145, "sum": 0.01284956756249145, "min": 0.01284956756249145}}, "EndTime": 1552274182.305587, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274182.305572}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012990787114929314, "sum": 0.012990787114929314, "min": 0.012990787114929314}}, "EndTime": 1552274182.305643, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274182.305628}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012850412979796904, "sum": 0.012850412979796904, "min": 0.012850412979796904}}, "EndTime": 1552274182.305697, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274182.305682}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.012991039510348334, "sum": 0.012991039510348334, "min": 0.012991039510348334}}, "EndTime": 1552274182.305752, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274182.305737}
    [0m
    [31m[03/11/2019 03:16:22 INFO 140227862767424] #quality_metric: host=algo-1, epoch=14, train binary_classification_cross_entropy_objective <loss>=0.00486511984542[0m
    [31m[03/11/2019 03:16:22 INFO 140227862767424] #early_stopping_criteria_metric: host=algo-1, epoch=14, criteria=binary_classification_cross_entropy_objective, value=0.00418340777108[0m
    [31m[03/11/2019 03:16:22 INFO 140227862767424] Epoch 14: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:16:22 INFO 140227862767424] #progress_metric: host=algo-1, completed 100 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 3012, "sum": 3012.0, "min": 3012}, "Total Records Seen": {"count": 1, "max": 3002460, "sum": 3002460.0, "min": 3002460}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 17, "sum": 17.0, "min": 17}}, "EndTime": 1552274182.3082, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274176.12319}
    [0m
    [31m[03/11/2019 03:16:22 INFO 140227862767424] #throughput_metric: host=algo-1, train throughput=32232.7750484 records/second[0m
    [31m[03/11/2019 03:16:22 WARNING 140227862767424] wait_for_all_workers will not sync workers since the kv store is not running distributed[0m
    [31m[03/11/2019 03:16:22 WARNING 140227862767424] wait_for_all_workers will not sync workers since the kv store is not running distributed[0m
    [31m[2019-03-11 03:16:22.308] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 16, "duration": 6185, "num_examples": 200}[0m
    [31m[2019-03-11 03:16:22.314] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 17, "duration": 5, "num_examples": 1}[0m
    [31m[2019-03-11 03:16:22.991] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 18, "duration": 675, "num_examples": 200}[0m
    [31m[03/11/2019 03:16:23 INFO 140227862767424] #train_score (algo-1) : ('binary_classification_cross_entropy_objective', 0.0041373788497844092)[0m
    [31m[03/11/2019 03:16:23 INFO 140227862767424] #train_score (algo-1) : ('binary_classification_accuracy', 0.96143235488854561)[0m
    [31m[03/11/2019 03:16:23 INFO 140227862767424] #train_score (algo-1) : ('binary_f_1.000', 0.07661822985468957)[0m
    [31m[03/11/2019 03:16:23 INFO 140227862767424] #train_score (algo-1) : ('precision', 0.039994984954864594)[0m
    [31m[03/11/2019 03:16:23 INFO 140227862767424] #train_score (algo-1) : ('recall', 0.9088319088319088)[0m
    [31m[03/11/2019 03:16:23 INFO 140227862767424] #quality_metric: host=algo-1, train binary_classification_cross_entropy_objective <loss>=0.00413737884978[0m
    [31m[03/11/2019 03:16:23 INFO 140227862767424] #quality_metric: host=algo-1, train binary_classification_accuracy <score>=0.961432354889[0m
    [31m[03/11/2019 03:16:23 INFO 140227862767424] #quality_metric: host=algo-1, train binary_f_1.000 <score>=0.0766182298547[0m
    [31m[03/11/2019 03:16:23 INFO 140227862767424] #quality_metric: host=algo-1, train precision <score>=0.0399949849549[0m
    [31m[03/11/2019 03:16:23 INFO 140227862767424] #quality_metric: host=algo-1, train recall <score>=0.908831908832[0m
    [31m[03/11/2019 03:16:23 INFO 140227862767424] Best model found for hyperparameters: {"lr_scheduler_step": 10, "wd": 0.0001, "optimizer": "adam", "lr_scheduler_factor": 0.99, "l1": 0.0, "learning_rate": 0.1, "lr_scheduler_minimum_lr": 0.0001}[0m
    [31m[03/11/2019 03:16:23 INFO 140227862767424] Saved checkpoint to "/tmp/tmpGWJVjj/mx-mod-0000.params"[0m
    [31m[03/11/2019 03:16:23 INFO 140227862767424] Test data is not provided.[0m
    [31m[2019-03-11 03:16:23.687] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 19, "duration": 696, "num_examples": 200}[0m
    [31m[2019-03-11 03:16:23.687] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "duration": 92443, "num_epochs": 20, "num_examples": 3413}[0m
    [31m#metrics {"Metrics": {"totaltime": {"count": 1, "max": 92645.98989486694, "sum": 92645.98989486694, "min": 92645.98989486694}, "finalize.time": {"count": 1, "max": 1371.2379932403564, "sum": 1371.2379932403564, "min": 1371.2379932403564}, "initialize.time": {"count": 1, "max": 171.49901390075684, "sum": 171.49901390075684, "min": 171.49901390075684}, "check_early_stopping.time": {"count": 15, "max": 0.8919239044189453, "sum": 12.119770050048828, "min": 0.7419586181640625}, "setuptime": {"count": 1, "max": 18.522977828979492, "sum": 18.522977828979492, "min": 18.522977828979492}, "update.time": {"count": 15, "max": 6388.322114944458, "sum": 90999.40514564514, "min": 5672.186851501465}, "epochs": {"count": 1, "max": 15, "sum": 15.0, "min": 15}}, "EndTime": 1552274183.688054, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1552274091.125929}
    [0m
    
    2019-03-11 03:16:33 Completed - Training job completed
    Billable seconds: 120
    CPU times: user 658 ms, sys: 53 ms, total: 711 ms
    Wall time: 5min 12s


### Deploy and evaluate the tuned estimator

Deploy the tuned predictor and evaluate it.

We hypothesized that a tuned model, optimized for a higher recall, would have fewer false negatives (fraudulent transactions incorrectly labeled as valid); did the number of false negatives get reduced after tuning the model?


```python
%%time 
# deploy and create a predictor
recall_predictor = linear_recall.deploy(initial_instance_count=1, instance_type='ml.t2.medium')
```

    INFO:sagemaker:Creating model with name: linear-learner-2019-03-11-03-17-13-236
    INFO:sagemaker:Creating endpoint with name linear-learner-2019-03-11-03-12-00-255


    --------------------------------------------------------------------------!CPU times: user 388 ms, sys: 9.17 ms, total: 397 ms
    Wall time: 6min 14s



```python
print('Metrics for tuned (recall), LinearLearner.\n')

# get metrics for tuned predictor
metrics = evaluate(recall_predictor, 
                   test_features.astype('float32'), 
                   test_labels, 
                   verbose=True)
```

    Metrics for tuned (recall), LinearLearner.
    
    prediction (col)    0.0   1.0
    actual (row)                 
    0.0               81913  3389
    1.0                  10   131
    
    Recall:     0.929
    Precision:  0.037
    Accuracy:   0.960
    


## Delete the endpoint 

As always, when you're done evaluating a model, you should delete the endpoint. Below, I'm using the `delete_endpoint` helper function I defined earlier.


```python
# delete the predictor endpoint 
delete_endpoint(recall_predictor)
```

    Deleted linear-learner-2019-03-11-03-12-00-255


---
## Improvement: Managing Class Imbalance

We have a model that is tuned to get a higher recall, which aims to reduce the number of false negatives. Earlier, we discussed how class imbalance may actually bias our model towards predicting that all transactions are valid, resulting in higher false negatives and true negatives. It stands to reason that this model could be further improved if we account for this imbalance.

To account for class imbalance during training of a binary classifier, LinearLearner offers the hyperparameter, `positive_example_weight_mult`, which is the weight assigned to positive (1, fraudulent) examples when training a binary classifier. The weight of negative examples (0, valid) is fixed at 1. 

### EXERCISE: Create a LinearLearner with a `positive_example_weight_mult` parameter

In **addition** to tuning a model for higher recall (you may use `linear_recall` as a starting point), you should *add* a parameter that helps account for class imbalance. From the [hyperparameter documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/ll_hyperparameters.html) on `positive_example_weight_mult`, it reads:
> "If you want the algorithm to choose a weight so that errors in classifying negative vs. positive examples have equal impact on training loss, specify `balanced`."

You could also put in a specific float value, in which case you'd want to weight positive examples more heavily than negative examples, since there are fewer of them.


```python
# instantiate a LinearLearner

# include params for tuning for higher recall
# *and* account for class imbalance in training data
linear_balanced = LinearLearner(role=role,
                                train_instance_count=1, 
                                train_instance_type='ml.c4.xlarge',
                                predictor_type='binary_classifier',
                                output_path=output_path,
                                sagemaker_session=sagemaker_session,
                                epochs=15,
                                binary_classifier_model_selection_criteria='precision_at_target_recall', # target recall
                                target_recall=0.9,
                                positive_example_weight_mult='balanced')

```

### EXERCISE: Train the balanced estimator

Fit the new, balanced estimator on the formatted training data.


```python
%%time 
# train the estimator on formatted training data
linear_balanced.fit(formatted_train_data)
```

    INFO:sagemaker:Creating training-job with name: linear-learner-2019-03-11-03-23-39-913


    2019-03-11 03:23:40 Starting - Starting the training job...
    2019-03-11 03:23:41 Starting - Launching requested ML instances......
    2019-03-11 03:24:43 Starting - Preparing the instances for training......
    2019-03-11 03:26:08 Downloading - Downloading input data
    2019-03-11 03:26:08 Training - Downloading the training image.
    [31mDocker entrypoint called with argument(s): train[0m
    [31m[03/11/2019 03:26:14 INFO 140701971167040] Reading default configuration from /opt/amazon/lib/python2.7/site-packages/algorithm/default-input.json: {u'loss_insensitivity': u'0.01', u'epochs': u'15', u'init_bias': u'0.0', u'lr_scheduler_factor': u'auto', u'num_calibration_samples': u'10000000', u'accuracy_top_k': u'3', u'_num_kv_servers': u'auto', u'use_bias': u'true', u'num_point_for_scaler': u'10000', u'_log_level': u'info', u'quantile': u'0.5', u'bias_lr_mult': u'auto', u'lr_scheduler_step': u'auto', u'init_method': u'uniform', u'init_sigma': u'0.01', u'lr_scheduler_minimum_lr': u'auto', u'target_recall': u'0.8', u'num_models': u'auto', u'early_stopping_patience': u'3', u'momentum': u'auto', u'unbias_label': u'auto', u'wd': u'auto', u'optimizer': u'auto', u'_tuning_objective_metric': u'', u'early_stopping_tolerance': u'0.001', u'learning_rate': u'auto', u'_kvstore': u'auto', u'normalize_data': u'true', u'binary_classifier_model_selection_criteria': u'accuracy', u'use_lr_scheduler': u'true', u'target_precision': u'0.8', u'unbias_data': u'auto', u'init_scale': u'0.07', u'bias_wd_mult': u'auto', u'f_beta': u'1.0', u'mini_batch_size': u'1000', u'huber_delta': u'1.0', u'num_classes': u'1', u'beta_1': u'auto', u'loss': u'auto', u'beta_2': u'auto', u'_enable_profiler': u'false', u'normalize_label': u'auto', u'_num_gpus': u'auto', u'balance_multiclass_weights': u'false', u'positive_example_weight_mult': u'1.0', u'l1': u'auto', u'margin': u'1.0'}[0m
    [31m[03/11/2019 03:26:14 INFO 140701971167040] Reading provided configuration from /opt/ml/input/config/hyperparameters.json: {u'predictor_type': u'binary_classifier', u'feature_dim': u'30', u'binary_classifier_model_selection_criteria': u'precision_at_target_recall', u'epochs': u'15', u'positive_example_weight_mult': u'balanced', u'target_recall': u'0.9', u'mini_batch_size': u'1000'}[0m
    [31m[03/11/2019 03:26:14 INFO 140701971167040] Final configuration: {u'loss_insensitivity': u'0.01', u'epochs': u'15', u'feature_dim': u'30', u'init_bias': u'0.0', u'lr_scheduler_factor': u'auto', u'num_calibration_samples': u'10000000', u'accuracy_top_k': u'3', u'_num_kv_servers': u'auto', u'use_bias': u'true', u'num_point_for_scaler': u'10000', u'_log_level': u'info', u'quantile': u'0.5', u'bias_lr_mult': u'auto', u'lr_scheduler_step': u'auto', u'init_method': u'uniform', u'init_sigma': u'0.01', u'lr_scheduler_minimum_lr': u'auto', u'target_recall': u'0.9', u'num_models': u'auto', u'early_stopping_patience': u'3', u'momentum': u'auto', u'unbias_label': u'auto', u'wd': u'auto', u'optimizer': u'auto', u'_tuning_objective_metric': u'', u'early_stopping_tolerance': u'0.001', u'learning_rate': u'auto', u'_kvstore': u'auto', u'normalize_data': u'true', u'binary_classifier_model_selection_criteria': u'precision_at_target_recall', u'use_lr_scheduler': u'true', u'target_precision': u'0.8', u'unbias_data': u'auto', u'init_scale': u'0.07', u'bias_wd_mult': u'auto', u'f_beta': u'1.0', u'mini_batch_size': u'1000', u'huber_delta': u'1.0', u'num_classes': u'1', u'predictor_type': u'binary_classifier', u'beta_1': u'auto', u'loss': u'auto', u'beta_2': u'auto', u'_enable_profiler': u'false', u'normalize_label': u'auto', u'_num_gpus': u'auto', u'balance_multiclass_weights': u'false', u'positive_example_weight_mult': u'balanced', u'l1': u'auto', u'margin': u'1.0'}[0m
    [31m[03/11/2019 03:26:14 WARNING 140701971167040] Loggers have already been setup.[0m
    [31mProcess 1 is a worker.[0m
    [31m[03/11/2019 03:26:14 INFO 140701971167040] Using default worker.[0m
    [31m[2019-03-11 03:26:14.877] [tensorio] [info] batch={"data_pipeline": "/opt/ml/input/data/train", "num_examples": 1000, "features": [{"name": "label_values", "shape": [1], "storage_type": "dense"}, {"name": "values", "shape": [30], "storage_type": "dense"}]}[0m
    [31m[2019-03-11 03:26:14.904] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 0, "duration": 27, "num_examples": 1}[0m
    [31m[03/11/2019 03:26:14 INFO 140701971167040] Create Store: local[0m
    [31m[2019-03-11 03:26:14.970] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 1, "duration": 64, "num_examples": 11}[0m
    [31m[03/11/2019 03:26:14 INFO 140701971167040] Scaler algorithm parameters
     <algorithm.scaler.ScalerAlgorithmStable object at 0x7ff766d38a50>[0m
    [31m[03/11/2019 03:26:14 INFO 140701971167040] Scaling model computed with parameters:
     {'stdev_weight': [0m
    [31m[  4.75497891e+04   2.01225400e+00   1.72936726e+00   1.48752689e+00
       1.41830683e+00   1.42959750e+00   1.34760964e+00   1.27067423e+00
       1.24293745e+00   1.09265101e+00   1.05321789e+00   1.01260686e+00
       9.87991810e-01   1.00782645e+00   9.47202206e-01   9.02963459e-01
       8.68877888e-01   8.27179432e-01   8.36477458e-01   8.07050884e-01
       8.00110519e-01   7.55493522e-01   7.21427202e-01   6.25614405e-01
       6.10876381e-01   5.16283095e-01   4.88118291e-01   4.35698181e-01
       3.69419903e-01   2.47155548e+02][0m
    [31m<NDArray 30 @cpu(0)>, 'stdev_label': None, 'mean_label': None, 'mean_weight': [0m
    [31m[  9.44802812e+04  -1.04726264e-02  -1.43008800e-02   1.28451567e-02
       1.87512934e-02  -2.48281248e-02   5.86199807e-03  -7.13069551e-03
      -7.39883492e-03   1.20382467e-02   6.10911567e-03  -3.16866231e-03
       8.64854374e-04   2.46435311e-03   1.56665407e-02   1.12619074e-02
      -4.91584092e-03  -1.56447978e-03   2.45723873e-03   2.82235094e-04
      -3.25949211e-03   6.57527940e-03   3.11945518e-03   6.22356636e-03
      -6.13171898e-04  -3.88089707e-03   1.16021503e-02  -3.21021304e-03
      -5.27510792e-03   8.94287567e+01][0m
    [31m<NDArray 30 @cpu(0)>}[0m
    [31m[03/11/2019 03:26:15 INFO 140701971167040] nvidia-smi took: 0.0251779556274 secs to identify 0 gpus[0m
    [31m[03/11/2019 03:26:15 INFO 140701971167040] Number of GPUs being used: 0[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 11, "sum": 11.0, "min": 11}, "Number of Batches Since Last Reset": {"count": 1, "max": 0, "sum": 0.0, "min": 0}, "Number of Records Since Last Reset": {"count": 1, "max": 0, "sum": 0.0, "min": 0}, "Total Batches Seen": {"count": 1, "max": 12, "sum": 12.0, "min": 12}, "Total Records Seen": {"count": 1, "max": 12000, "sum": 12000.0, "min": 12000}, "Max Records Seen Between Resets": {"count": 1, "max": 11000, "sum": 11000.0, "min": 11000}, "Reset Count": {"count": 1, "max": 2, "sum": 2.0, "min": 2}}, "EndTime": 1552274775.130212, "Dimensions": {"Host": "algo-1", "Meta": "init_train_data_iter", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1552274775.130173}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6594284648799417, "sum": 0.6594284648799417, "min": 0.6594284648799417}}, "EndTime": 1552274781.132004, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274781.131933}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5641238785940199, "sum": 0.5641238785940199, "min": 0.5641238785940199}}, "EndTime": 1552274781.13209, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274781.132077}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5921783411994053, "sum": 0.5921783411994053, "min": 0.5921783411994053}}, "EndTime": 1552274781.13219, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274781.132168}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6301838542228967, "sum": 0.6301838542228967, "min": 0.6301838542228967}}, "EndTime": 1552274781.132257, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274781.13224}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6843703128297125, "sum": 0.6843703128297125, "min": 0.6843703128297125}}, "EndTime": 1552274781.132349, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274781.132301}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6846481458385986, "sum": 0.6846481458385986, "min": 0.6846481458385986}}, "EndTime": 1552274781.132395, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274781.132385}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6839004336313985, "sum": 0.6839004336313985, "min": 0.6839004336313985}}, "EndTime": 1552274781.132424, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274781.132417}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.7122602933567374, "sum": 0.7122602933567374, "min": 0.7122602933567374}}, "EndTime": 1552274781.132451, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274781.132444}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5813303722592454, "sum": 0.5813303722592454, "min": 0.5813303722592454}}, "EndTime": 1552274781.13251, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274781.132494}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6624351184020689, "sum": 0.6624351184020689, "min": 0.6624351184020689}}, "EndTime": 1552274781.132569, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274781.132552}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6498399631366059, "sum": 0.6498399631366059, "min": 0.6498399631366059}}, "EndTime": 1552274781.132626, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274781.13261}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6519365340189718, "sum": 0.6519365340189718, "min": 0.6519365340189718}}, "EndTime": 1552274781.132681, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274781.132665}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6812790213350075, "sum": 0.6812790213350075, "min": 0.6812790213350075}}, "EndTime": 1552274781.132735, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274781.13272}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6948909245591667, "sum": 0.6948909245591667, "min": 0.6948909245591667}}, "EndTime": 1552274781.132829, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274781.132812}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6716007738161327, "sum": 0.6716007738161327, "min": 0.6716007738161327}}, "EndTime": 1552274781.132884, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274781.132869}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6938190375476626, "sum": 0.6938190375476626, "min": 0.6938190375476626}}, "EndTime": 1552274781.13294, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274781.132924}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6940059118989724, "sum": 0.6940059118989724, "min": 0.6940059118989724}}, "EndTime": 1552274781.132994, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274781.132978}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5978640082277844, "sum": 0.5978640082277844, "min": 0.5978640082277844}}, "EndTime": 1552274781.133046, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274781.133031}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6243885920538974, "sum": 0.6243885920538974, "min": 0.6243885920538974}}, "EndTime": 1552274781.133102, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274781.133085}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6476971526026126, "sum": 0.6476971526026126, "min": 0.6476971526026126}}, "EndTime": 1552274781.13316, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274781.133144}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.63529909008352, "sum": 0.63529909008352, "min": 0.63529909008352}}, "EndTime": 1552274781.133213, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274781.133198}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6430660022563072, "sum": 0.6430660022563072, "min": 0.6430660022563072}}, "EndTime": 1552274781.133265, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274781.133251}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6422417014999007, "sum": 0.6422417014999007, "min": 0.6422417014999007}}, "EndTime": 1552274781.133319, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274781.133303}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6510663895918496, "sum": 0.6510663895918496, "min": 0.6510663895918496}}, "EndTime": 1552274781.133373, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274781.133357}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1656365715295227, "sum": 1.1656365715295227, "min": 1.1656365715295227}}, "EndTime": 1552274781.133427, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274781.133412}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1654303487653108, "sum": 1.1654303487653108, "min": 1.1654303487653108}}, "EndTime": 1552274781.133484, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274781.133469}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1970537133911747, "sum": 1.1970537133911747, "min": 1.1970537133911747}}, "EndTime": 1552274781.13354, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274781.133525}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.171867919921875, "sum": 1.171867919921875, "min": 1.171867919921875}}, "EndTime": 1552274781.133596, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274781.133581}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.3268262262392283, "sum": 1.3268262262392283, "min": 1.3268262262392283}}, "EndTime": 1552274781.133648, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274781.133634}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.3632041089235238, "sum": 1.3632041089235238, "min": 1.3632041089235238}}, "EndTime": 1552274781.133704, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274781.133689}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.332283373348677, "sum": 1.332283373348677, "min": 1.332283373348677}}, "EndTime": 1552274781.133769, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274781.133752}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.3859594543303677, "sum": 1.3859594543303677, "min": 1.3859594543303677}}, "EndTime": 1552274781.133824, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274781.133809}
    [0m
    [31m[03/11/2019 03:26:21 INFO 140701971167040] #quality_metric: host=algo-1, epoch=0, train binary_classification_weighted_cross_entropy_objective <loss>=0.65942846488[0m
    [31m[03/11/2019 03:26:21 INFO 140701971167040] #early_stopping_criteria_metric: host=algo-1, epoch=0, criteria=binary_classification_weighted_cross_entropy_objective, value=0.564123878594[0m
    [31m[03/11/2019 03:26:21 INFO 140701971167040] Epoch 0: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:26:21 INFO 140701971167040] #progress_metric: host=algo-1, completed 6 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 212, "sum": 212.0, "min": 212}, "Total Records Seen": {"count": 1, "max": 211364, "sum": 211364.0, "min": 211364}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 3, "sum": 3.0, "min": 3}}, "EndTime": 1552274781.137, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552274775.130432}
    [0m
    [31m[03/11/2019 03:26:21 INFO 140701971167040] #throughput_metric: host=algo-1, train throughput=33190.3694539 records/second[0m
    [31m[2019-03-11 03:26:21.137] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 2, "duration": 6006, "num_examples": 200}[0m
    
    2019-03-11 03:26:12 Training - Training image download completed. Training in progress.[31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.4400108514526981, "sum": 0.4400108514526981, "min": 0.4400108514526981}}, "EndTime": 1552274788.06281, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274788.062722}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.41052518957703554, "sum": 0.41052518957703554, "min": 0.41052518957703554}}, "EndTime": 1552274788.062902, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274788.062888}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.4252546185057367, "sum": 0.4252546185057367, "min": 0.4252546185057367}}, "EndTime": 1552274788.062939, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274788.06293}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.42712178238911847, "sum": 0.42712178238911847, "min": 0.42712178238911847}}, "EndTime": 1552274788.062972, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274788.062964}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5716547131178966, "sum": 0.5716547131178966, "min": 0.5716547131178966}}, "EndTime": 1552274788.063019, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274788.063004}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.670325926584215, "sum": 0.670325926584215, "min": 0.670325926584215}}, "EndTime": 1552274788.063061, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274788.063047}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5771199898647903, "sum": 0.5771199898647903, "min": 0.5771199898647903}}, "EndTime": 1552274788.06311, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274788.063097}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6394495888714814, "sum": 0.6394495888714814, "min": 0.6394495888714814}}, "EndTime": 1552274788.063164, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274788.06315}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.4237853919561185, "sum": 0.4237853919561185, "min": 0.4237853919561185}}, "EndTime": 1552274788.063225, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274788.063209}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.43772098046690977, "sum": 0.43772098046690977, "min": 0.43772098046690977}}, "EndTime": 1552274788.063284, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274788.063267}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.4373177432726376, "sum": 0.4373177432726376, "min": 0.4373177432726376}}, "EndTime": 1552274788.063352, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274788.063336}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.4355643713102868, "sum": 0.4355643713102868, "min": 0.4355643713102868}}, "EndTime": 1552274788.063406, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274788.063391}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5609734526016006, "sum": 0.5609734526016006, "min": 0.5609734526016006}}, "EndTime": 1552274788.063459, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274788.063444}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6396466855763191, "sum": 0.6396466855763191, "min": 0.6396466855763191}}, "EndTime": 1552274788.063511, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274788.063497}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5679037587534842, "sum": 0.5679037587534842, "min": 0.5679037587534842}}, "EndTime": 1552274788.063564, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274788.063549}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6381025119378938, "sum": 0.6381025119378938, "min": 0.6381025119378938}}, "EndTime": 1552274788.063619, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274788.063604}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5228390950725306, "sum": 0.5228390950725306, "min": 0.5228390950725306}}, "EndTime": 1552274788.063672, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274788.063658}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5129850728403983, "sum": 0.5129850728403983, "min": 0.5129850728403983}}, "EndTime": 1552274788.063724, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274788.06371}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5143203467747673, "sum": 0.5143203467747673, "min": 0.5143203467747673}}, "EndTime": 1552274788.063787, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274788.063771}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5161900682305571, "sum": 0.5161900682305571, "min": 0.5161900682305571}}, "EndTime": 1552274788.063852, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274788.063835}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5769340392452987, "sum": 0.5769340392452987, "min": 0.5769340392452987}}, "EndTime": 1552274788.06391, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274788.063894}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6002559302056855, "sum": 0.6002559302056855, "min": 0.6002559302056855}}, "EndTime": 1552274788.063964, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274788.063949}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5767329220028978, "sum": 0.5767329220028978, "min": 0.5767329220028978}}, "EndTime": 1552274788.064027, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274788.064011}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6016254009074302, "sum": 0.6016254009074302, "min": 0.6016254009074302}}, "EndTime": 1552274788.064083, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274788.064068}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1656406513770021, "sum": 1.1656406513770021, "min": 1.1656406513770021}}, "EndTime": 1552274788.064137, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274788.064123}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1657563387616794, "sum": 1.1657563387616794, "min": 1.1657563387616794}}, "EndTime": 1552274788.064194, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274788.064179}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1665071908001923, "sum": 1.1665071908001923, "min": 1.1665071908001923}}, "EndTime": 1552274788.064258, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274788.064241}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1657475589004593, "sum": 1.1657475589004593, "min": 1.1657475589004593}}, "EndTime": 1552274788.064313, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274788.064299}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.268502646748145, "sum": 1.268502646748145, "min": 1.268502646748145}}, "EndTime": 1552274788.064369, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274788.064354}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.3813459926586056, "sum": 1.3813459926586056, "min": 1.3813459926586056}}, "EndTime": 1552274788.064425, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274788.064409}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.2631105847382664, "sum": 1.2631105847382664, "min": 1.2631105847382664}}, "EndTime": 1552274788.064487, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274788.064471}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.3678211852413924, "sum": 1.3678211852413924, "min": 1.3678211852413924}}, "EndTime": 1552274788.064543, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274788.064528}
    [0m
    [31m[03/11/2019 03:26:28 INFO 140701971167040] #quality_metric: host=algo-1, epoch=1, train binary_classification_weighted_cross_entropy_objective <loss>=0.440010851453[0m
    [31m[03/11/2019 03:26:28 INFO 140701971167040] #early_stopping_criteria_metric: host=algo-1, epoch=1, criteria=binary_classification_weighted_cross_entropy_objective, value=0.410525189577[0m
    [31m[03/11/2019 03:26:28 INFO 140701971167040] Epoch 1: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:26:28 INFO 140701971167040] #progress_metric: host=algo-1, completed 13 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 412, "sum": 412.0, "min": 412}, "Total Records Seen": {"count": 1, "max": 410728, "sum": 410728.0, "min": 410728}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 4, "sum": 4.0, "min": 4}}, "EndTime": 1552274788.067283, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552274781.13731}
    [0m
    [31m[03/11/2019 03:26:28 INFO 140701971167040] #throughput_metric: host=algo-1, train throughput=28767.8221717 records/second[0m
    [31m[2019-03-11 03:26:28.067] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 3, "duration": 6930, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.39636924766655546, "sum": 0.39636924766655546, "min": 0.39636924766655546}}, "EndTime": 1552274794.266855, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274794.266789}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.376585278074945, "sum": 0.376585278074945, "min": 0.376585278074945}}, "EndTime": 1552274794.266945, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274794.266928}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.38781511101650834, "sum": 0.38781511101650834, "min": 0.38781511101650834}}, "EndTime": 1552274794.267002, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274794.266987}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.38641533173987613, "sum": 0.38641533173987613, "min": 0.38641533173987613}}, "EndTime": 1552274794.267054, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274794.267039}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5263680824394801, "sum": 0.5263680824394801, "min": 0.5263680824394801}}, "EndTime": 1552274794.267104, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274794.267091}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6410529355571498, "sum": 0.6410529355571498, "min": 0.6410529355571498}}, "EndTime": 1552274794.267156, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274794.267141}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.527668864609608, "sum": 0.527668864609608, "min": 0.527668864609608}}, "EndTime": 1552274794.267211, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274794.267195}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6855561453852821, "sum": 0.6855561453852821, "min": 0.6855561453852821}}, "EndTime": 1552274794.267263, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274794.267248}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.38759695419234846, "sum": 0.38759695419234846, "min": 0.38759695419234846}}, "EndTime": 1552274794.267315, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274794.2673}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.39293417833797895, "sum": 0.39293417833797895, "min": 0.39293417833797895}}, "EndTime": 1552274794.267371, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274794.267355}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3962465733667115, "sum": 0.3962465733667115, "min": 0.3962465733667115}}, "EndTime": 1552274794.267427, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274794.267411}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.39143112677185976, "sum": 0.39143112677185976, "min": 0.39143112677185976}}, "EndTime": 1552274794.267481, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274794.267466}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5372402820491311, "sum": 0.5372402820491311, "min": 0.5372402820491311}}, "EndTime": 1552274794.267535, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274794.26752}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6481121471347522, "sum": 0.6481121471347522, "min": 0.6481121471347522}}, "EndTime": 1552274794.267599, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274794.267574}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5270041325631453, "sum": 0.5270041325631453, "min": 0.5270041325631453}}, "EndTime": 1552274794.267651, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274794.267636}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6470744394657001, "sum": 0.6470744394657001, "min": 0.6470744394657001}}, "EndTime": 1552274794.267705, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274794.26769}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5103085941333867, "sum": 0.5103085941333867, "min": 0.5103085941333867}}, "EndTime": 1552274794.267758, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274794.267744}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5106808740050349, "sum": 0.5106808740050349, "min": 0.5106808740050349}}, "EndTime": 1552274794.267812, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274794.267797}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5085814014990725, "sum": 0.5085814014990725, "min": 0.5085814014990725}}, "EndTime": 1552274794.267864, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274794.267849}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5110635287030857, "sum": 0.5110635287030857, "min": 0.5110635287030857}}, "EndTime": 1552274794.267916, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274794.267902}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.565871770734164, "sum": 0.565871770734164, "min": 0.565871770734164}}, "EndTime": 1552274794.267967, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274794.267953}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5999275546625031, "sum": 0.5999275546625031, "min": 0.5999275546625031}}, "EndTime": 1552274794.26802, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274794.268005}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.566059082146266, "sum": 0.566059082146266, "min": 0.566059082146266}}, "EndTime": 1552274794.268072, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274794.268057}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5999852474921912, "sum": 0.5999852474921912, "min": 0.5999852474921912}}, "EndTime": 1552274794.268126, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274794.268111}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.166252090530779, "sum": 1.166252090530779, "min": 1.166252090530779}}, "EndTime": 1552274794.268178, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274794.268163}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1656566413611025, "sum": 1.1656566413611025, "min": 1.1656566413611025}}, "EndTime": 1552274794.268233, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274794.268218}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1664664858717415, "sum": 1.1664664858717415, "min": 1.1664664858717415}}, "EndTime": 1552274794.268279, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274794.268269}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.165680956969908, "sum": 1.165680956969908, "min": 1.165680956969908}}, "EndTime": 1552274794.268306, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274794.268299}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.2412207515083964, "sum": 1.2412207515083964, "min": 1.2412207515083964}}, "EndTime": 1552274794.268356, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274794.268341}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.360423379332576, "sum": 1.360423379332576, "min": 1.360423379332576}}, "EndTime": 1552274794.268412, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274794.268396}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.2389932212637897, "sum": 1.2389932212637897, "min": 1.2389932212637897}}, "EndTime": 1552274794.268465, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274794.26845}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.356852158666256, "sum": 1.356852158666256, "min": 1.356852158666256}}, "EndTime": 1552274794.268522, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274794.268506}
    [0m
    [31m[03/11/2019 03:26:34 INFO 140701971167040] #quality_metric: host=algo-1, epoch=2, train binary_classification_weighted_cross_entropy_objective <loss>=0.396369247667[0m
    [31m[03/11/2019 03:26:34 INFO 140701971167040] #early_stopping_criteria_metric: host=algo-1, epoch=2, criteria=binary_classification_weighted_cross_entropy_objective, value=0.376585278075[0m
    [31m[03/11/2019 03:26:34 INFO 140701971167040] Epoch 2: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:26:34 INFO 140701971167040] #progress_metric: host=algo-1, completed 20 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 612, "sum": 612.0, "min": 612}, "Total Records Seen": {"count": 1, "max": 610092, "sum": 610092.0, "min": 610092}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 5, "sum": 5.0, "min": 5}}, "EndTime": 1552274794.271185, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552274788.067602}
    [0m
    [31m[03/11/2019 03:26:34 INFO 140701971167040] #throughput_metric: host=algo-1, train throughput=32136.2794771 records/second[0m
    [31m[2019-03-11 03:26:34.271] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 4, "duration": 6203, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3764211182714108, "sum": 0.3764211182714108, "min": 0.3764211182714108}}, "EndTime": 1552274800.35278, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274800.352667}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3615751494211168, "sum": 0.3615751494211168, "min": 0.3615751494211168}}, "EndTime": 1552274800.352884, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274800.352868}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3704841190510659, "sum": 0.3704841190510659, "min": 0.3704841190510659}}, "EndTime": 1552274800.35293, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274800.352919}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3682524187097597, "sum": 0.3682524187097597, "min": 0.3682524187097597}}, "EndTime": 1552274800.35297, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274800.35296}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.497972018179582, "sum": 0.497972018179582, "min": 0.497972018179582}}, "EndTime": 1552274800.353008, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274800.352998}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.658230786692557, "sum": 0.658230786692557, "min": 0.658230786692557}}, "EndTime": 1552274800.353045, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274800.353035}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.4991150675634643, "sum": 0.4991150675634643, "min": 0.4991150675634643}}, "EndTime": 1552274800.35308, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274800.353071}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6169741138285728, "sum": 0.6169741138285728, "min": 0.6169741138285728}}, "EndTime": 1552274800.353117, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274800.353107}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3709429301352956, "sum": 0.3709429301352956, "min": 0.3709429301352956}}, "EndTime": 1552274800.353153, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274800.353144}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3725585856222028, "sum": 0.3725585856222028, "min": 0.3725585856222028}}, "EndTime": 1552274800.353189, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274800.353179}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.37727097776786767, "sum": 0.37727097776786767, "min": 0.37727097776786767}}, "EndTime": 1552274800.353225, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274800.353216}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.37159369559743294, "sum": 0.37159369559743294, "min": 0.37159369559743294}}, "EndTime": 1552274800.353262, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274800.353253}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5094950629766263, "sum": 0.5094950629766263, "min": 0.5094950629766263}}, "EndTime": 1552274800.353298, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274800.353289}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6470716767239211, "sum": 0.6470716767239211, "min": 0.6470716767239211}}, "EndTime": 1552274800.353333, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274800.353324}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5039542786487982, "sum": 0.5039542786487982, "min": 0.5039542786487982}}, "EndTime": 1552274800.353368, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274800.353359}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6462183145877703, "sum": 0.6462183145877703, "min": 0.6462183145877703}}, "EndTime": 1552274800.353404, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274800.353395}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5069897746943949, "sum": 0.5069897746943949, "min": 0.5069897746943949}}, "EndTime": 1552274800.353448, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274800.35344}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5102628221368071, "sum": 0.5102628221368071, "min": 0.5102628221368071}}, "EndTime": 1552274800.353483, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274800.353474}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5066128204192348, "sum": 0.5066128204192348, "min": 0.5066128204192348}}, "EndTime": 1552274800.353517, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274800.353508}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5103834318994876, "sum": 0.5103834318994876, "min": 0.5103834318994876}}, "EndTime": 1552274800.353554, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274800.353545}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5548706218393604, "sum": 0.5548706218393604, "min": 0.5548706218393604}}, "EndTime": 1552274800.353589, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274800.353581}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5986631354902258, "sum": 0.5986631354902258, "min": 0.5986631354902258}}, "EndTime": 1552274800.353624, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274800.353615}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.554863652444964, "sum": 0.554863652444964, "min": 0.554863652444964}}, "EndTime": 1552274800.353659, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274800.35365}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5985999432664421, "sum": 0.5985999432664421, "min": 0.5985999432664421}}, "EndTime": 1552274800.353693, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274800.353684}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.166377805163513, "sum": 1.166377805163513, "min": 1.166377805163513}}, "EndTime": 1552274800.353727, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274800.353718}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1656277021091788, "sum": 1.1656277021091788, "min": 1.1656277021091788}}, "EndTime": 1552274800.353762, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274800.353753}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1663435764025205, "sum": 1.1663435764025205, "min": 1.1663435764025205}}, "EndTime": 1552274800.353796, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274800.353788}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1656497630977152, "sum": 1.1656497630977152, "min": 1.1656497630977152}}, "EndTime": 1552274800.353831, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274800.353822}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.226151385896769, "sum": 1.226151385896769, "min": 1.226151385896769}}, "EndTime": 1552274800.353865, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274800.353856}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.3499431781097873, "sum": 1.3499431781097873, "min": 1.3499431781097873}}, "EndTime": 1552274800.3539, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274800.353891}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.2239721780901578, "sum": 1.2239721780901578, "min": 1.2239721780901578}}, "EndTime": 1552274800.353934, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274800.353926}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.348765112948777, "sum": 1.348765112948777, "min": 1.348765112948777}}, "EndTime": 1552274800.353971, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274800.353962}
    [0m
    [31m[03/11/2019 03:26:40 INFO 140701971167040] #quality_metric: host=algo-1, epoch=3, train binary_classification_weighted_cross_entropy_objective <loss>=0.376421118271[0m
    [31m[03/11/2019 03:26:40 INFO 140701971167040] #early_stopping_criteria_metric: host=algo-1, epoch=3, criteria=binary_classification_weighted_cross_entropy_objective, value=0.361575149421[0m
    [31m[03/11/2019 03:26:40 INFO 140701971167040] Epoch 3: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:26:40 INFO 140701971167040] #progress_metric: host=algo-1, completed 26 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 812, "sum": 812.0, "min": 812}, "Total Records Seen": {"count": 1, "max": 809456, "sum": 809456.0, "min": 809456}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 6, "sum": 6.0, "min": 6}}, "EndTime": 1552274800.356684, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552274794.271474}
    [0m
    [31m[03/11/2019 03:26:40 INFO 140701971167040] #throughput_metric: host=algo-1, train throughput=32761.1957712 records/second[0m
    [31m[2019-03-11 03:26:40.356] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 5, "duration": 6085, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3651828865549672, "sum": 0.3651828865549672, "min": 0.3651828865549672}}, "EndTime": 1552274806.321253, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274806.321187}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3535009060193546, "sum": 0.3535009060193546, "min": 0.3535009060193546}}, "EndTime": 1552274806.321337, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274806.321324}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.36066419469172034, "sum": 0.36066419469172034, "min": 0.36066419469172034}}, "EndTime": 1552274806.321371, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274806.321363}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3583622215309335, "sum": 0.3583622215309335, "min": 0.3583622215309335}}, "EndTime": 1552274806.321403, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274806.321395}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.48255323714826576, "sum": 0.48255323714826576, "min": 0.48255323714826576}}, "EndTime": 1552274806.321431, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274806.321424}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6404641866636036, "sum": 0.6404641866636036, "min": 0.6404641866636036}}, "EndTime": 1552274806.321457, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274806.32145}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.48250414187944113, "sum": 0.48250414187944113, "min": 0.48250414187944113}}, "EndTime": 1552274806.321483, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274806.321476}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6370406882127925, "sum": 0.6370406882127925, "min": 0.6370406882127925}}, "EndTime": 1552274806.321516, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274806.321503}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.36155209388924603, "sum": 0.36155209388924603, "min": 0.36155209388924603}}, "EndTime": 1552274806.321563, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274806.321549}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.36159848973259856, "sum": 0.36159848973259856, "min": 0.36159848973259856}}, "EndTime": 1552274806.32161, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274806.321597}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3665388703466061, "sum": 0.3665388703466061, "min": 0.3665388703466061}}, "EndTime": 1552274806.321662, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274806.321648}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3610365468700927, "sum": 0.3610365468700927, "min": 0.3610365468700927}}, "EndTime": 1552274806.321715, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274806.3217}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.43677961008273175, "sum": 0.43677961008273175, "min": 0.43677961008273175}}, "EndTime": 1552274806.321768, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274806.321753}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6353890059964741, "sum": 0.6353890059964741, "min": 0.6353890059964741}}, "EndTime": 1552274806.321822, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274806.321807}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.48457892844425376, "sum": 0.48457892844425376, "min": 0.48457892844425376}}, "EndTime": 1552274806.321875, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274806.32186}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6465855449695683, "sum": 0.6465855449695683, "min": 0.6465855449695683}}, "EndTime": 1552274806.321931, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274806.321915}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5052605467369808, "sum": 0.5052605467369808, "min": 0.5052605467369808}}, "EndTime": 1552274806.321987, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274806.321971}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5100626200767019, "sum": 0.5100626200767019, "min": 0.5100626200767019}}, "EndTime": 1552274806.322051, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274806.322036}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5052216142050585, "sum": 0.5052216142050585, "min": 0.5052216142050585}}, "EndTime": 1552274806.322103, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274806.322088}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5101361088585015, "sum": 0.5101361088585015, "min": 0.5101361088585015}}, "EndTime": 1552274806.322155, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274806.322141}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5432823336040554, "sum": 0.5432823336040554, "min": 0.5432823336040554}}, "EndTime": 1552274806.322208, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274806.322193}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5973346039278423, "sum": 0.5973346039278423, "min": 0.5973346039278423}}, "EndTime": 1552274806.322261, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274806.322246}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5431448340104453, "sum": 0.5431448340104453, "min": 0.5431448340104453}}, "EndTime": 1552274806.322315, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274806.3223}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5972607800852713, "sum": 0.5972607800852713, "min": 0.5972607800852713}}, "EndTime": 1552274806.322369, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274806.322354}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1660935969520454, "sum": 1.1660935969520454, "min": 1.1660935969520454}}, "EndTime": 1552274806.322422, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274806.322407}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1656119823359963, "sum": 1.1656119823359963, "min": 1.1656119823359963}}, "EndTime": 1552274806.322476, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274806.322461}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1659982738399026, "sum": 1.1659982738399026, "min": 1.1659982738399026}}, "EndTime": 1552274806.32253, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274806.322514}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.165628966058319, "sum": 1.165628966058319, "min": 1.165628966058319}}, "EndTime": 1552274806.322594, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274806.322578}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.2179688801981097, "sum": 1.2179688801981097, "min": 1.2179688801981097}}, "EndTime": 1552274806.322651, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274806.322635}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.3417915374142442, "sum": 1.3417915374142442, "min": 1.3417915374142442}}, "EndTime": 1552274806.322705, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274806.32269}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.2180785441182966, "sum": 1.2180785441182966, "min": 1.2180785441182966}}, "EndTime": 1552274806.322767, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274806.322751}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.3412464424785058, "sum": 1.3412464424785058, "min": 1.3412464424785058}}, "EndTime": 1552274806.322822, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274806.322807}
    [0m
    [31m[03/11/2019 03:26:46 INFO 140701971167040] #quality_metric: host=algo-1, epoch=4, train binary_classification_weighted_cross_entropy_objective <loss>=0.365182886555[0m
    [31m[03/11/2019 03:26:46 INFO 140701971167040] #early_stopping_criteria_metric: host=algo-1, epoch=4, criteria=binary_classification_weighted_cross_entropy_objective, value=0.353500906019[0m
    [31m[03/11/2019 03:26:46 INFO 140701971167040] Epoch 4: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:26:46 INFO 140701971167040] #progress_metric: host=algo-1, completed 33 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 1012, "sum": 1012.0, "min": 1012}, "Total Records Seen": {"count": 1, "max": 1008820, "sum": 1008820.0, "min": 1008820}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 7, "sum": 7.0, "min": 7}}, "EndTime": 1552274806.325387, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552274800.357027}
    [0m
    [31m[03/11/2019 03:26:46 INFO 140701971167040] #throughput_metric: host=algo-1, train throughput=33402.8101433 records/second[0m
    [31m[2019-03-11 03:26:46.325] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 6, "duration": 5968, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.35806176627461034, "sum": 0.35806176627461034, "min": 0.35806176627461034}}, "EndTime": 1552274813.168993, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274813.168918}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34855323484794576, "sum": 0.34855323484794576, "min": 0.34855323484794576}}, "EndTime": 1552274813.169086, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274813.169068}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3543773801794004, "sum": 0.3543773801794004, "min": 0.3543773801794004}}, "EndTime": 1552274813.169149, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274813.169133}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3522900390625, "sum": 0.3522900390625, "min": 0.3522900390625}}, "EndTime": 1552274813.169206, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274813.16919}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.41642863010042275, "sum": 0.41642863010042275, "min": 0.41642863010042275}}, "EndTime": 1552274813.169277, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274813.169259}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6387309703731058, "sum": 0.6387309703731058, "min": 0.6387309703731058}}, "EndTime": 1552274813.169334, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274813.169318}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.4263452541792213, "sum": 0.4263452541792213, "min": 0.4263452541792213}}, "EndTime": 1552274813.169393, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274813.169376}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6099737069786494, "sum": 0.6099737069786494, "min": 0.6099737069786494}}, "EndTime": 1552274813.169458, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274813.169441}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.35555598238844366, "sum": 0.35555598238844366, "min": 0.35555598238844366}}, "EndTime": 1552274813.16952, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274813.169504}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.355062516869013, "sum": 0.355062516869013, "min": 0.355062516869013}}, "EndTime": 1552274813.169579, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274813.169564}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.35968572714340746, "sum": 0.35968572714340746, "min": 0.35968572714340746}}, "EndTime": 1552274813.169645, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274813.169619}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.35472683363104585, "sum": 0.35472683363104585, "min": 0.35472683363104585}}, "EndTime": 1552274813.169701, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274813.169686}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.4209054723384991, "sum": 0.4209054723384991, "min": 0.4209054723384991}}, "EndTime": 1552274813.169757, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274813.169741}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6355837141257434, "sum": 0.6355837141257434, "min": 0.6355837141257434}}, "EndTime": 1552274813.169811, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274813.169796}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.4188486150425283, "sum": 0.4188486150425283, "min": 0.4188486150425283}}, "EndTime": 1552274813.169867, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274813.169852}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6536027886280462, "sum": 0.6536027886280462, "min": 0.6536027886280462}}, "EndTime": 1552274813.169922, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274813.169907}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.503968337475954, "sum": 0.503968337475954, "min": 0.503968337475954}}, "EndTime": 1552274813.169976, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274813.16996}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5099149558675948, "sum": 0.5099149558675948, "min": 0.5099149558675948}}, "EndTime": 1552274813.170022, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274813.170009}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5040184928855704, "sum": 0.5040184928855704, "min": 0.5040184928855704}}, "EndTime": 1552274813.170068, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274813.170054}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5099665906896543, "sum": 0.5099665906896543, "min": 0.5099665906896543}}, "EndTime": 1552274813.170119, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274813.170105}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5343341753015566, "sum": 0.5343341753015566, "min": 0.5343341753015566}}, "EndTime": 1552274813.170173, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274813.170159}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5959916668034079, "sum": 0.5959916668034079, "min": 0.5959916668034079}}, "EndTime": 1552274813.170224, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274813.17021}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5342394232055051, "sum": 0.5342394232055051, "min": 0.5342394232055051}}, "EndTime": 1552274813.170287, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274813.170271}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5959276057487756, "sum": 0.5959276057487756, "min": 0.5959276057487756}}, "EndTime": 1552274813.170343, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274813.170328}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.165481177861966, "sum": 1.165481177861966, "min": 1.165481177861966}}, "EndTime": 1552274813.170397, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274813.170382}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.165601423253965, "sum": 1.165601423253965, "min": 1.165601423253965}}, "EndTime": 1552274813.170445, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274813.170435}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1653950100232608, "sum": 1.1653950100232608, "min": 1.1653950100232608}}, "EndTime": 1552274813.170494, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274813.17048}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1656148482279562, "sum": 1.1656148482279562, "min": 1.1656148482279562}}, "EndTime": 1552274813.170549, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274813.170538}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1937079349690347, "sum": 1.1937079349690347, "min": 1.1937079349690347}}, "EndTime": 1552274813.170594, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274813.17058}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.3341801171230911, "sum": 1.3341801171230911, "min": 1.3341801171230911}}, "EndTime": 1552274813.170646, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274813.170632}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1937398098892902, "sum": 1.1937398098892902, "min": 1.1937398098892902}}, "EndTime": 1552274813.170701, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274813.170686}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.3338634862851857, "sum": 1.3338634862851857, "min": 1.3338634862851857}}, "EndTime": 1552274813.170735, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274813.170727}
    [0m
    [31m[03/11/2019 03:26:53 INFO 140701971167040] #quality_metric: host=algo-1, epoch=5, train binary_classification_weighted_cross_entropy_objective <loss>=0.358061766275[0m
    [31m[03/11/2019 03:26:53 INFO 140701971167040] #early_stopping_criteria_metric: host=algo-1, epoch=5, criteria=binary_classification_weighted_cross_entropy_objective, value=0.348553234848[0m
    [31m[03/11/2019 03:26:53 INFO 140701971167040] Epoch 5: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:26:53 INFO 140701971167040] #progress_metric: host=algo-1, completed 40 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 1212, "sum": 1212.0, "min": 1212}, "Total Records Seen": {"count": 1, "max": 1208184, "sum": 1208184.0, "min": 1208184}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 8, "sum": 8.0, "min": 8}}, "EndTime": 1552274813.173473, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552274806.325687}
    [0m
    [31m[03/11/2019 03:26:53 INFO 140701971167040] #throughput_metric: host=algo-1, train throughput=29113.107451 records/second[0m
    [31m[2019-03-11 03:26:53.173] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 7, "duration": 6847, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.35317101517873795, "sum": 0.35317101517873795, "min": 0.35317101517873795}}, "EndTime": 1552274819.786792, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274819.786701}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34528309504830057, "sum": 0.34528309504830057, "min": 0.34528309504830057}}, "EndTime": 1552274819.786896, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274819.78688}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3500301142936975, "sum": 0.3500301142936975, "min": 0.3500301142936975}}, "EndTime": 1552274819.786956, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274819.786943}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34819938180434645, "sum": 0.34819938180434645, "min": 0.34819938180434645}}, "EndTime": 1552274819.787005, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274819.786994}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3891295891766572, "sum": 0.3891295891766572, "min": 0.3891295891766572}}, "EndTime": 1552274819.78705, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274819.78704}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6296487351614027, "sum": 0.6296487351614027, "min": 0.6296487351614027}}, "EndTime": 1552274819.787091, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274819.787081}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.394590725731011, "sum": 0.394590725731011, "min": 0.394590725731011}}, "EndTime": 1552274819.787134, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274819.787124}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6279934317814045, "sum": 0.6279934317814045, "min": 0.6279934317814045}}, "EndTime": 1552274819.787172, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274819.787163}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.35141688480089656, "sum": 0.35141688480089656, "min": 0.35141688480089656}}, "EndTime": 1552274819.787214, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274819.787205}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.35082043813580843, "sum": 0.35082043813580843, "min": 0.35082043813580843}}, "EndTime": 1552274819.787266, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274819.787255}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.35497230564289955, "sum": 0.35497230564289955, "min": 0.35497230564289955}}, "EndTime": 1552274819.787308, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274819.787298}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3506037244940523, "sum": 0.3506037244940523, "min": 0.3506037244940523}}, "EndTime": 1552274819.787347, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274819.787338}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.39505894408872977, "sum": 0.39505894408872977, "min": 0.39505894408872977}}, "EndTime": 1552274819.78739, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274819.78738}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6260122773443634, "sum": 0.6260122773443634, "min": 0.6260122773443634}}, "EndTime": 1552274819.787443, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274819.787423}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.4093167758155708, "sum": 0.4093167758155708, "min": 0.4093167758155708}}, "EndTime": 1552274819.78748, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274819.787471}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6923932212177832, "sum": 0.6923932212177832, "min": 0.6923932212177832}}, "EndTime": 1552274819.787522, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274819.787513}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5028794399242306, "sum": 0.5028794399242306, "min": 0.5028794399242306}}, "EndTime": 1552274819.787564, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274819.787554}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5097846717259392, "sum": 0.5097846717259392, "min": 0.5097846717259392}}, "EndTime": 1552274819.787602, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274819.787593}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5029439985572394, "sum": 0.5029439985572394, "min": 0.5029439985572394}}, "EndTime": 1552274819.78764, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274819.787631}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5098219938805355, "sum": 0.5098219938805355, "min": 0.5098219938805355}}, "EndTime": 1552274819.787681, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274819.787671}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5277016542520955, "sum": 0.5277016542520955, "min": 0.5277016542520955}}, "EndTime": 1552274819.787726, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274819.787715}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.594609693920193, "sum": 0.594609693920193, "min": 0.594609693920193}}, "EndTime": 1552274819.78777, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274819.78776}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5276506366825583, "sum": 0.5276506366825583, "min": 0.5276506366825583}}, "EndTime": 1552274819.787809, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274819.787799}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5945276401845654, "sum": 0.5945276401845654, "min": 0.5945276401845654}}, "EndTime": 1552274819.787846, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274819.787836}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1646659637911236, "sum": 1.1646659637911236, "min": 1.1646659637911236}}, "EndTime": 1552274819.787887, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274819.787877}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1655957068055118, "sum": 1.1655957068055118, "min": 1.1655957068055118}}, "EndTime": 1552274819.787932, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274819.787921}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.164603540775165, "sum": 1.164603540775165, "min": 1.164603540775165}}, "EndTime": 1552274819.787975, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274819.787965}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1656060886095516, "sum": 1.1656060886095516, "min": 1.1656060886095516}}, "EndTime": 1552274819.788016, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274819.788004}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1840833206560144, "sum": 1.1840833206560144, "min": 1.1840833206560144}}, "EndTime": 1552274819.788051, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274819.788038}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.3269847950384246, "sum": 1.3269847950384246, "min": 1.3269847950384246}}, "EndTime": 1552274819.78809, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274819.788082}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1839885206366305, "sum": 1.1839885206366305, "min": 1.1839885206366305}}, "EndTime": 1552274819.788121, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274819.78811}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.3267813469201477, "sum": 1.3267813469201477, "min": 1.3267813469201477}}, "EndTime": 1552274819.788173, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274819.78816}
    [0m
    [31m[03/11/2019 03:26:59 INFO 140701971167040] #quality_metric: host=algo-1, epoch=6, train binary_classification_weighted_cross_entropy_objective <loss>=0.353171015179[0m
    [31m[03/11/2019 03:26:59 INFO 140701971167040] #early_stopping_criteria_metric: host=algo-1, epoch=6, criteria=binary_classification_weighted_cross_entropy_objective, value=0.345283095048[0m
    [31m[03/11/2019 03:26:59 INFO 140701971167040] Epoch 6: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:26:59 INFO 140701971167040] #progress_metric: host=algo-1, completed 46 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 1412, "sum": 1412.0, "min": 1412}, "Total Records Seen": {"count": 1, "max": 1407548, "sum": 1407548.0, "min": 1407548}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 9, "sum": 9.0, "min": 9}}, "EndTime": 1552274819.790775, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552274813.173764}
    [0m
    [31m[03/11/2019 03:26:59 INFO 140701971167040] #throughput_metric: host=algo-1, train throughput=30128.401017 records/second[0m
    [31m[2019-03-11 03:26:59.791] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 8, "duration": 6617, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.349640905178971, "sum": 0.349640905178971, "min": 0.349640905178971}}, "EndTime": 1552274826.033999, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274826.033932}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34294272801384856, "sum": 0.34294272801384856, "min": 0.34294272801384856}}, "EndTime": 1552274826.034079, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274826.034066}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34686880205624065, "sum": 0.34686880205624065, "min": 0.34686880205624065}}, "EndTime": 1552274826.034143, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274826.034126}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34525132471592584, "sum": 0.34525132471592584, "min": 0.34525132471592584}}, "EndTime": 1552274826.034202, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274826.034186}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.39339666817056473, "sum": 0.39339666817056473, "min": 0.39339666817056473}}, "EndTime": 1552274826.03426, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274826.034244}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6260022582816119, "sum": 0.6260022582816119, "min": 0.6260022582816119}}, "EndTime": 1552274826.034317, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274826.034301}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3888993842733565, "sum": 0.3888993842733565, "min": 0.3888993842733565}}, "EndTime": 1552274826.034372, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274826.034356}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6040140958239685, "sum": 0.6040140958239685, "min": 0.6040140958239685}}, "EndTime": 1552274826.034424, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274826.034409}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3484091053871653, "sum": 0.3484091053871653, "min": 0.3484091053871653}}, "EndTime": 1552274826.034489, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274826.034471}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.347946627784614, "sum": 0.347946627784614, "min": 0.347946627784614}}, "EndTime": 1552274826.034555, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274826.034539}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.35155894040821783, "sum": 0.35155894040821783, "min": 0.35155894040821783}}, "EndTime": 1552274826.034619, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274826.034603}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3477796048495039, "sum": 0.3477796048495039, "min": 0.3477796048495039}}, "EndTime": 1552274826.034682, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274826.034666}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3969215943226263, "sum": 0.3969215943226263, "min": 0.3969215943226263}}, "EndTime": 1552274826.034745, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274826.034729}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6151315656307355, "sum": 0.6151315656307355, "min": 0.6151315656307355}}, "EndTime": 1552274826.034807, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274826.034791}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3832322796002105, "sum": 0.3832322796002105, "min": 0.3832322796002105}}, "EndTime": 1552274826.034878, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274826.034863}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6553429662618205, "sum": 0.6553429662618205, "min": 0.6553429662618205}}, "EndTime": 1552274826.034938, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274826.034923}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5019372299424368, "sum": 0.5019372299424368, "min": 0.5019372299424368}}, "EndTime": 1552274826.034987, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274826.034975}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5096615561480499, "sum": 0.5096615561480499, "min": 0.5096615561480499}}, "EndTime": 1552274826.035034, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274826.035022}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5019936022734522, "sum": 0.5019936022734522, "min": 0.5019936022734522}}, "EndTime": 1552274826.035082, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274826.035069}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5096888131759874, "sum": 0.5096888131759874, "min": 0.5096888131759874}}, "EndTime": 1552274826.035136, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274826.035121}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5229926883563324, "sum": 0.5229926883563324, "min": 0.5229926883563324}}, "EndTime": 1552274826.035188, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274826.035174}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5930843838447303, "sum": 0.5930843838447303, "min": 0.5930843838447303}}, "EndTime": 1552274826.03524, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274826.035225}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5229712625628141, "sum": 0.5229712625628141, "min": 0.5229712625628141}}, "EndTime": 1552274826.035291, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274826.035278}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5930094521872362, "sum": 0.5930094521872362, "min": 0.5930094521872362}}, "EndTime": 1552274826.035344, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274826.035329}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1639128795221223, "sum": 1.1639128795221223, "min": 1.1639128795221223}}, "EndTime": 1552274826.035396, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274826.035381}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1655941846071176, "sum": 1.1655941846071176, "min": 1.1655941846071176}}, "EndTime": 1552274826.03545, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274826.035435}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1638703628616716, "sum": 1.1638703628616716, "min": 1.1638703628616716}}, "EndTime": 1552274826.035502, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274826.035488}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.16560186368856, "sum": 1.16560186368856, "min": 1.16560186368856}}, "EndTime": 1552274826.03556, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274826.035544}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1810852133592769, "sum": 1.1810852133592769, "min": 1.1810852133592769}}, "EndTime": 1552274826.03562, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274826.035604}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.32025075182124, "sum": 1.32025075182124, "min": 1.32025075182124}}, "EndTime": 1552274826.03567, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274826.035655}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1810465575558455, "sum": 1.1810465575558455, "min": 1.1810465575558455}}, "EndTime": 1552274826.035728, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274826.035713}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.3201223543253375, "sum": 1.3201223543253375, "min": 1.3201223543253375}}, "EndTime": 1552274826.035782, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274826.035767}
    [0m
    [31m[03/11/2019 03:27:06 INFO 140701971167040] #quality_metric: host=algo-1, epoch=7, train binary_classification_weighted_cross_entropy_objective <loss>=0.349640905179[0m
    [31m[03/11/2019 03:27:06 INFO 140701971167040] #early_stopping_criteria_metric: host=algo-1, epoch=7, criteria=binary_classification_weighted_cross_entropy_objective, value=0.342942728014[0m
    [31m[03/11/2019 03:27:06 INFO 140701971167040] Epoch 7: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:27:06 INFO 140701971167040] #progress_metric: host=algo-1, completed 53 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 1612, "sum": 1612.0, "min": 1612}, "Total Records Seen": {"count": 1, "max": 1606912, "sum": 1606912.0, "min": 1606912}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 10, "sum": 10.0, "min": 10}}, "EndTime": 1552274826.038304, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552274819.79108}
    [0m
    [31m[03/11/2019 03:27:06 INFO 140701971167040] #throughput_metric: host=algo-1, train throughput=31911.8011521 records/second[0m
    [31m[2019-03-11 03:27:06.038] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 9, "duration": 6247, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.347006891145179, "sum": 0.347006891145179, "min": 0.347006891145179}}, "EndTime": 1552274832.677393, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274832.677328}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3411967257782442, "sum": 0.3411967257782442, "min": 0.3411967257782442}}, "EndTime": 1552274832.677483, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274832.677464}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3444969539163101, "sum": 0.3444969539163101, "min": 0.3444969539163101}}, "EndTime": 1552274832.677539, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274832.677524}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34306995453187566, "sum": 0.34306995453187566, "min": 0.34306995453187566}}, "EndTime": 1552274832.677594, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274832.677579}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.37010197816781665, "sum": 0.37010197816781665, "min": 0.37010197816781665}}, "EndTime": 1552274832.677643, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274832.677629}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6173712888746405, "sum": 0.6173712888746405, "min": 0.6173712888746405}}, "EndTime": 1552274832.677696, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274832.677681}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3686550999550364, "sum": 0.3686550999550364, "min": 0.3686550999550364}}, "EndTime": 1552274832.677752, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274832.677737}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.619885236979729, "sum": 0.619885236979729, "min": 0.619885236979729}}, "EndTime": 1552274832.6778, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274832.677788}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3461459124004422, "sum": 0.3461459124004422, "min": 0.3461459124004422}}, "EndTime": 1552274832.67787, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274832.677853}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3458885993765826, "sum": 0.3458885993765826, "min": 0.3458885993765826}}, "EndTime": 1552274832.67793, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274832.677914}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3490008349394679, "sum": 0.3490008349394679, "min": 0.3490008349394679}}, "EndTime": 1552274832.677985, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274832.67797}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.345774281947457, "sum": 0.345774281947457, "min": 0.345774281947457}}, "EndTime": 1552274832.678042, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274832.678027}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3744585236209122, "sum": 0.3744585236209122, "min": 0.3744585236209122}}, "EndTime": 1552274832.678097, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274832.678083}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6302096776147583, "sum": 0.6302096776147583, "min": 0.6302096776147583}}, "EndTime": 1552274832.678165, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274832.67814}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3754779644299991, "sum": 0.3754779644299991, "min": 0.3754779644299991}}, "EndTime": 1552274832.67822, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274832.678206}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5735211174931358, "sum": 0.5735211174931358, "min": 0.5735211174931358}}, "EndTime": 1552274832.678273, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274832.678259}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5011246051117404, "sum": 0.5011246051117404, "min": 0.5011246051117404}}, "EndTime": 1552274832.678336, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274832.67832}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5095432196382302, "sum": 0.5095432196382302, "min": 0.5095432196382302}}, "EndTime": 1552274832.678397, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274832.678382}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5011678960598893, "sum": 0.5011678960598893, "min": 0.5011678960598893}}, "EndTime": 1552274832.67845, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274832.678435}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5095630672589019, "sum": 0.5095630672589019, "min": 0.5095630672589019}}, "EndTime": 1552274832.678512, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274832.678496}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.51962202921345, "sum": 0.51962202921345, "min": 0.51962202921345}}, "EndTime": 1552274832.678575, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274832.678559}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5914428989851295, "sum": 0.5914428989851295, "min": 0.5914428989851295}}, "EndTime": 1552274832.678628, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274832.678613}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5196154634101906, "sum": 0.5196154634101906, "min": 0.5196154634101906}}, "EndTime": 1552274832.678681, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274832.678666}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5913778166651127, "sum": 0.5913778166651127, "min": 0.5913778166651127}}, "EndTime": 1552274832.678741, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274832.678726}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.163333273422778, "sum": 1.163333273422778, "min": 1.163333273422778}}, "EndTime": 1552274832.678803, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274832.678788}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.165600453132361, "sum": 1.165600453132361, "min": 1.165600453132361}}, "EndTime": 1552274832.678866, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274832.67885}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.16330359224099, "sum": 1.16330359224099, "min": 1.16330359224099}}, "EndTime": 1552274832.678929, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274832.678912}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1656058300535883, "sum": 1.1656058300535883, "min": 1.1656058300535883}}, "EndTime": 1552274832.678985, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274832.678972}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1798437187156485, "sum": 1.1798437187156485, "min": 1.1798437187156485}}, "EndTime": 1552274832.679044, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274832.679029}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.313886191900052, "sum": 1.313886191900052, "min": 1.313886191900052}}, "EndTime": 1552274832.679096, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274832.679082}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.179719946013024, "sum": 1.179719946013024, "min": 1.179719946013024}}, "EndTime": 1552274832.679149, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274832.679134}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.313814590070715, "sum": 1.313814590070715, "min": 1.313814590070715}}, "EndTime": 1552274832.679191, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274832.679178}
    [0m
    [31m[03/11/2019 03:27:12 INFO 140701971167040] #quality_metric: host=algo-1, epoch=8, train binary_classification_weighted_cross_entropy_objective <loss>=0.347006891145[0m
    [31m[03/11/2019 03:27:12 INFO 140701971167040] #early_stopping_criteria_metric: host=algo-1, epoch=8, criteria=binary_classification_weighted_cross_entropy_objective, value=0.341196725778[0m
    [31m[03/11/2019 03:27:12 INFO 140701971167040] Epoch 8: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:27:12 INFO 140701971167040] #progress_metric: host=algo-1, completed 60 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 1812, "sum": 1812.0, "min": 1812}, "Total Records Seen": {"count": 1, "max": 1806276, "sum": 1806276.0, "min": 1806276}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 11, "sum": 11.0, "min": 11}}, "EndTime": 1552274832.681805, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552274826.038605}
    [0m
    [31m[03/11/2019 03:27:12 INFO 140701971167040] #throughput_metric: host=algo-1, train throughput=30009.6623375 records/second[0m
    [31m[2019-03-11 03:27:12.682] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 10, "duration": 6643, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34497469671048114, "sum": 0.34497469671048114, "min": 0.34497469671048114}}, "EndTime": 1552274839.262245, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274839.262177}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.33982736052700024, "sum": 0.33982736052700024, "min": 0.33982736052700024}}, "EndTime": 1552274839.262328, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274839.262315}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3426661689413253, "sum": 0.3426661689413253, "min": 0.3426661689413253}}, "EndTime": 1552274839.26238, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274839.262367}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3414100283138716, "sum": 0.3414100283138716, "min": 0.3414100283138716}}, "EndTime": 1552274839.262433, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274839.262419}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3568473606588852, "sum": 0.3568473606588852, "min": 0.3568473606588852}}, "EndTime": 1552274839.262466, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274839.262457}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6148977909183981, "sum": 0.6148977909183981, "min": 0.6148977909183981}}, "EndTime": 1552274839.262514, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274839.2625}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3576055860663179, "sum": 0.3576055860663179, "min": 0.3576055860663179}}, "EndTime": 1552274839.262565, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274839.262551}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5979439923463754, "sum": 0.5979439923463754, "min": 0.5979439923463754}}, "EndTime": 1552274839.262617, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274839.262602}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3444105252596601, "sum": 0.3444105252596601, "min": 0.3444105252596601}}, "EndTime": 1552274839.262651, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274839.262643}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3443532883917267, "sum": 0.3443532883917267, "min": 0.3443532883917267}}, "EndTime": 1552274839.262687, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274839.262674}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3470415938487604, "sum": 0.3470415938487604, "min": 0.3470415938487604}}, "EndTime": 1552274839.262738, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274839.262724}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3442744213085079, "sum": 0.3442744213085079, "min": 0.3442744213085079}}, "EndTime": 1552274839.262791, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274839.262776}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.36758389485421494, "sum": 0.36758389485421494, "min": 0.36758389485421494}}, "EndTime": 1552274839.262843, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274839.26283}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6228373950018955, "sum": 0.6228373950018955, "min": 0.6228373950018955}}, "EndTime": 1552274839.262895, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274839.26288}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3621376801303883, "sum": 0.3621376801303883, "min": 0.3621376801303883}}, "EndTime": 1552274839.262963, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274839.262947}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6296973451010546, "sum": 0.6296973451010546, "min": 0.6296973451010546}}, "EndTime": 1552274839.26302, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274839.263005}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.500431356075421, "sum": 0.500431356075421, "min": 0.500431356075421}}, "EndTime": 1552274839.263076, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274839.263061}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5094262746686312, "sum": 0.5094262746686312, "min": 0.5094262746686312}}, "EndTime": 1552274839.263132, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274839.263117}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.500462592426856, "sum": 0.500462592426856, "min": 0.500462592426856}}, "EndTime": 1552274839.263185, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274839.263171}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5094407391572119, "sum": 0.5094407391572119, "min": 0.5094407391572119}}, "EndTime": 1552274839.26324, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274839.263225}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5170382586148516, "sum": 0.5170382586148516, "min": 0.5170382586148516}}, "EndTime": 1552274839.263292, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274839.263279}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5896801722924314, "sum": 0.5896801722924314, "min": 0.5896801722924314}}, "EndTime": 1552274839.263322, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274839.263314}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5170357649146612, "sum": 0.5170357649146612, "min": 0.5170357649146612}}, "EndTime": 1552274839.263371, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274839.263356}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5896412415624264, "sum": 0.5896412415624264, "min": 0.5896412415624264}}, "EndTime": 1552274839.263427, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274839.263411}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1628935074542635, "sum": 1.1628935074542635, "min": 1.1628935074542635}}, "EndTime": 1552274839.26348, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274839.263465}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1656102248915476, "sum": 1.1656102248915476, "min": 1.1656102248915476}}, "EndTime": 1552274839.263534, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274839.263519}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1628731707855684, "sum": 1.1628731707855684, "min": 1.1628731707855684}}, "EndTime": 1552274839.26359, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274839.263575}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.165613682425801, "sum": 1.165613682425801, "min": 1.165613682425801}}, "EndTime": 1552274839.263641, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274839.26363}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1754669865747194, "sum": 1.1754669865747194, "min": 1.1754669865747194}}, "EndTime": 1552274839.263669, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274839.263662}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.3078792055216266, "sum": 1.3078792055216266, "min": 1.3078792055216266}}, "EndTime": 1552274839.263694, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274839.263688}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1753525574650596, "sum": 1.1753525574650596, "min": 1.1753525574650596}}, "EndTime": 1552274839.263719, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274839.263712}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.3078432888625255, "sum": 1.3078432888625255, "min": 1.3078432888625255}}, "EndTime": 1552274839.263743, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274839.263737}
    [0m
    [31m[03/11/2019 03:27:19 INFO 140701971167040] #quality_metric: host=algo-1, epoch=9, train binary_classification_weighted_cross_entropy_objective <loss>=0.34497469671[0m
    [31m[03/11/2019 03:27:19 INFO 140701971167040] #early_stopping_criteria_metric: host=algo-1, epoch=9, criteria=binary_classification_weighted_cross_entropy_objective, value=0.339827360527[0m
    [31m[03/11/2019 03:27:19 INFO 140701971167040] Epoch 9: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:27:19 INFO 140701971167040] #progress_metric: host=algo-1, completed 66 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 2012, "sum": 2012.0, "min": 2012}, "Total Records Seen": {"count": 1, "max": 2005640, "sum": 2005640.0, "min": 2005640}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 12, "sum": 12.0, "min": 12}}, "EndTime": 1552274839.266419, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552274832.682078}
    [0m
    [31m[03/11/2019 03:27:19 INFO 140701971167040] #throughput_metric: host=algo-1, train throughput=30277.929993 records/second[0m
    [31m[2019-03-11 03:27:19.266] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 11, "duration": 6584, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.343385628781726, "sum": 0.343385628781726, "min": 0.343385628781726}}, "EndTime": 1552274845.812326, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274845.812234}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.33867450073855604, "sum": 0.33867450073855604, "min": 0.33867450073855604}}, "EndTime": 1552274845.812429, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274845.812413}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34123276504439926, "sum": 0.34123276504439926, "min": 0.34123276504439926}}, "EndTime": 1552274845.812483, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274845.812471}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34010360775281434, "sum": 0.34010360775281434, "min": 0.34010360775281434}}, "EndTime": 1552274845.81253, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274845.812519}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3513394770789985, "sum": 0.3513394770789985, "min": 0.3513394770789985}}, "EndTime": 1552274845.812573, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274845.812563}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6082766550342042, "sum": 0.6082766550342042, "min": 0.6082766550342042}}, "EndTime": 1552274845.812611, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274845.812601}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.35022275608388626, "sum": 0.35022275608388626, "min": 0.35022275608388626}}, "EndTime": 1552274845.812648, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274845.812639}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6111685697085893, "sum": 0.6111685697085893, "min": 0.6111685697085893}}, "EndTime": 1552274845.812685, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274845.812675}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34304809462964236, "sum": 0.34304809462964236, "min": 0.34304809462964236}}, "EndTime": 1552274845.812722, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274845.812712}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3431856944021867, "sum": 0.3431856944021867, "min": 0.3431856944021867}}, "EndTime": 1552274845.812794, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274845.812781}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3455205237997237, "sum": 0.3455205237997237, "min": 0.3455205237997237}}, "EndTime": 1552274845.812834, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274845.812825}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.343107326948463, "sum": 0.343107326948463, "min": 0.343107326948463}}, "EndTime": 1552274845.812872, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274845.812862}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.35804022151621145, "sum": 0.35804022151621145, "min": 0.35804022151621145}}, "EndTime": 1552274845.812908, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274845.812899}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5973278189232601, "sum": 0.5973278189232601, "min": 0.5973278189232601}}, "EndTime": 1552274845.812955, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274845.812936}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.35920516699402777, "sum": 0.35920516699402777, "min": 0.35920516699402777}}, "EndTime": 1552274845.81299, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274845.812981}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.553969459744554, "sum": 0.553969459744554, "min": 0.553969459744554}}, "EndTime": 1552274845.813029, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274845.813019}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.4998431644152157, "sum": 0.4998431644152157, "min": 0.4998431644152157}}, "EndTime": 1552274845.813064, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274845.813055}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5093097843956109, "sum": 0.5093097843956109, "min": 0.5093097843956109}}, "EndTime": 1552274845.813099, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274845.81309}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.499864671716738, "sum": 0.499864671716738, "min": 0.499864671716738}}, "EndTime": 1552274845.813134, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274845.813125}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5093203556693379, "sum": 0.5093203556693379, "min": 0.5093203556693379}}, "EndTime": 1552274845.813169, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274845.81316}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5148661216850856, "sum": 0.5148661216850856, "min": 0.5148661216850856}}, "EndTime": 1552274845.813204, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274845.813195}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5879017156859738, "sum": 0.5879017156859738, "min": 0.5879017156859738}}, "EndTime": 1552274845.813239, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274845.81323}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5148665229375637, "sum": 0.5148665229375637, "min": 0.5148665229375637}}, "EndTime": 1552274845.813275, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274845.813266}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5878681464075444, "sum": 0.5878681464075444, "min": 0.5878681464075444}}, "EndTime": 1552274845.81331, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274845.813302}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1625382359183614, "sum": 1.1625382359183614, "min": 1.1625382359183614}}, "EndTime": 1552274845.813345, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274845.813336}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1656226803573533, "sum": 1.1656226803573533, "min": 1.1656226803573533}}, "EndTime": 1552274845.813383, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274845.813373}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1625255948934123, "sum": 1.1625255948934123, "min": 1.1625255948934123}}, "EndTime": 1552274845.813419, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274845.81341}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1656246040382576, "sum": 1.1656246040382576, "min": 1.1656246040382576}}, "EndTime": 1552274845.813454, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274845.813445}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1713371868804472, "sum": 1.1713371868804472, "min": 1.1713371868804472}}, "EndTime": 1552274845.813507, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274845.813492}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.3022406913718985, "sum": 1.3022406913718985, "min": 1.3022406913718985}}, "EndTime": 1552274845.813566, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274845.813551}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.17130506168538, "sum": 1.17130506168538, "min": 1.17130506168538}}, "EndTime": 1552274845.81362, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274845.813605}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.3022192376678314, "sum": 1.3022192376678314, "min": 1.3022192376678314}}, "EndTime": 1552274845.813668, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274845.813654}
    [0m
    [31m[03/11/2019 03:27:25 INFO 140701971167040] #quality_metric: host=algo-1, epoch=10, train binary_classification_weighted_cross_entropy_objective <loss>=0.343385628782[0m
    [31m[03/11/2019 03:27:25 INFO 140701971167040] #early_stopping_criteria_metric: host=algo-1, epoch=10, criteria=binary_classification_weighted_cross_entropy_objective, value=0.338674500739[0m
    [31m[03/11/2019 03:27:25 INFO 140701971167040] Epoch 10: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:27:25 INFO 140701971167040] #progress_metric: host=algo-1, completed 73 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 2212, "sum": 2212.0, "min": 2212}, "Total Records Seen": {"count": 1, "max": 2205004, "sum": 2205004.0, "min": 2205004}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 13, "sum": 13.0, "min": 13}}, "EndTime": 1552274845.816253, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552274839.266699}
    [0m
    [31m[03/11/2019 03:27:25 INFO 140701971167040] #throughput_metric: host=algo-1, train throughput=30438.7157487 records/second[0m
    [31m[2019-03-11 03:27:25.816] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 12, "duration": 6549, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34214417086653975, "sum": 0.34214417086653975, "min": 0.34214417086653975}}, "EndTime": 1552274851.800038, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274851.799972}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3377189544601057, "sum": 0.3377189544601057, "min": 0.3377189544601057}}, "EndTime": 1552274851.80012, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274851.800107}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.340102334410701, "sum": 0.340102334410701, "min": 0.340102334410701}}, "EndTime": 1552274851.800173, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274851.80016}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3389972463636542, "sum": 0.3389972463636542, "min": 0.3389972463636542}}, "EndTime": 1552274851.800219, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274851.80021}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34368074595389053, "sum": 0.34368074595389053, "min": 0.34368074595389053}}, "EndTime": 1552274851.800269, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274851.800258}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6051267347096199, "sum": 0.6051267347096199, "min": 0.6051267347096199}}, "EndTime": 1552274851.800315, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274851.8003}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34431217971878436, "sum": 0.34431217971878436, "min": 0.34431217971878436}}, "EndTime": 1552274851.800369, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274851.800354}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5922150619545175, "sum": 0.5922150619545175, "min": 0.5922150619545175}}, "EndTime": 1552274851.800417, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274851.800404}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3419869154733629, "sum": 0.3419869154733629, "min": 0.3419869154733629}}, "EndTime": 1552274851.800471, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274851.800454}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3422508027733271, "sum": 0.3422508027733271, "min": 0.3422508027733271}}, "EndTime": 1552274851.800525, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274851.80051}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3443068614557161, "sum": 0.3443068614557161, "min": 0.3443068614557161}}, "EndTime": 1552274851.800577, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274851.800563}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3421717679967832, "sum": 0.3421717679967832, "min": 0.3421717679967832}}, "EndTime": 1552274851.800631, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274851.800617}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.35510520057103145, "sum": 0.35510520057103145, "min": 0.35510520057103145}}, "EndTime": 1552274851.800687, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274851.800671}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.588998402753667, "sum": 0.588998402753667, "min": 0.588998402753667}}, "EndTime": 1552274851.800763, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274851.800728}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.35120845434294273, "sum": 0.35120845434294273, "min": 0.35120845434294273}}, "EndTime": 1552274851.800833, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274851.800817}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.628038518435991, "sum": 0.628038518435991, "min": 0.628038518435991}}, "EndTime": 1552274851.800888, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274851.800873}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.4993475785758627, "sum": 0.4993475785758627, "min": 0.4993475785758627}}, "EndTime": 1552274851.800941, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274851.800926}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5091943125509137, "sum": 0.5091943125509137, "min": 0.5091943125509137}}, "EndTime": 1552274851.800996, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274851.800981}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.499361797236917, "sum": 0.499361797236917, "min": 0.499361797236917}}, "EndTime": 1552274851.801051, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274851.801036}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5092020252170275, "sum": 0.5092020252170275, "min": 0.5092020252170275}}, "EndTime": 1552274851.801105, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274851.80109}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5131318281164121, "sum": 0.5131318281164121, "min": 0.5131318281164121}}, "EndTime": 1552274851.801156, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274851.80114}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5861035508198954, "sum": 0.5861035508198954, "min": 0.5861035508198954}}, "EndTime": 1552274851.801206, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274851.801193}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5131352119637494, "sum": 0.5131352119637494, "min": 0.5131352119637494}}, "EndTime": 1552274851.801262, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274851.801248}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.586079047044917, "sum": 0.586079047044917, "min": 0.586079047044917}}, "EndTime": 1552274851.801317, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274851.801302}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1622375295054372, "sum": 1.1622375295054372, "min": 1.1622375295054372}}, "EndTime": 1552274851.801371, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274851.801356}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1656412175624216, "sum": 1.1656412175624216, "min": 1.1656412175624216}}, "EndTime": 1552274851.801426, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274851.801411}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1622304077148438, "sum": 1.1622304077148438, "min": 1.1622304077148438}}, "EndTime": 1552274851.801482, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274851.801467}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1656419435434007, "sum": 1.1656419435434007, "min": 1.1656419435434007}}, "EndTime": 1552274851.801544, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274851.801528}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1690166780864772, "sum": 1.1690166780864772, "min": 1.1690166780864772}}, "EndTime": 1552274851.801605, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274851.801589}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.2969777703213332, "sum": 1.2969777703213332, "min": 1.2969777703213332}}, "EndTime": 1552274851.801654, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274851.801644}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1689940323566073, "sum": 1.1689940323566073, "min": 1.1689940323566073}}, "EndTime": 1552274851.801692, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274851.801679}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.2969597775253219, "sum": 1.2969597775253219, "min": 1.2969597775253219}}, "EndTime": 1552274851.801753, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274851.801738}
    [0m
    [31m[03/11/2019 03:27:31 INFO 140701971167040] #quality_metric: host=algo-1, epoch=11, train binary_classification_weighted_cross_entropy_objective <loss>=0.342144170867[0m
    [31m[03/11/2019 03:27:31 INFO 140701971167040] #early_stopping_criteria_metric: host=algo-1, epoch=11, criteria=binary_classification_weighted_cross_entropy_objective, value=0.33771895446[0m
    [31m[03/11/2019 03:27:31 INFO 140701971167040] Epoch 11: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:27:31 INFO 140701971167040] #progress_metric: host=algo-1, completed 80 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 2412, "sum": 2412.0, "min": 2412}, "Total Records Seen": {"count": 1, "max": 2404368, "sum": 2404368.0, "min": 2404368}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 14, "sum": 14.0, "min": 14}}, "EndTime": 1552274851.804473, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552274845.816553}
    [0m
    [31m[03/11/2019 03:27:31 INFO 140701971167040] #throughput_metric: host=algo-1, train throughput=33293.6048191 records/second[0m
    [31m[2019-03-11 03:27:31.804] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 13, "duration": 5988, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34114156809284457, "sum": 0.34114156809284457, "min": 0.34114156809284457}}, "EndTime": 1552274857.243156, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274857.24309}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3368815071451005, "sum": 0.3368815071451005, "min": 0.3368815071451005}}, "EndTime": 1552274857.243237, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274857.243224}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.33919448994392126, "sum": 0.33919448994392126, "min": 0.33919448994392126}}, "EndTime": 1552274857.243278, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274857.243264}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3380386825638201, "sum": 0.3380386825638201, "min": 0.3380386825638201}}, "EndTime": 1552274857.243332, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274857.243317}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.33985611253287923, "sum": 0.33985611253287923, "min": 0.33985611253287923}}, "EndTime": 1552274857.243383, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274857.243369}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.598893346757745, "sum": 0.598893346757745, "min": 0.598893346757745}}, "EndTime": 1552274857.243436, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274857.243421}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3390653486395601, "sum": 0.3390653486395601, "min": 0.3390653486395601}}, "EndTime": 1552274857.243493, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274857.243477}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6011157977041887, "sum": 0.6011157977041887, "min": 0.6011157977041887}}, "EndTime": 1552274857.243549, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274857.243533}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34112556994260856, "sum": 0.34112556994260856, "min": 0.34112556994260856}}, "EndTime": 1552274857.243605, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274857.243589}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.341461145142215, "sum": 0.341461145142215, "min": 0.341461145142215}}, "EndTime": 1552274857.243671, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274857.243654}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3433391585613615, "sum": 0.3433391585613615, "min": 0.3433391585613615}}, "EndTime": 1552274857.243736, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274857.24372}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3414039370666197, "sum": 0.3414039370666197, "min": 0.3414039370666197}}, "EndTime": 1552274857.243799, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274857.243783}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34829548959396595, "sum": 0.34829548959396595, "min": 0.34829548959396595}}, "EndTime": 1552274857.24386, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274857.243845}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5974588073270405, "sum": 0.5974588073270405, "min": 0.5974588073270405}}, "EndTime": 1552274857.243933, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274857.243917}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34793724305426055, "sum": 0.34793724305426055, "min": 0.34793724305426055}}, "EndTime": 1552274857.243994, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274857.243979}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5641578885945843, "sum": 0.5641578885945843, "min": 0.5641578885945843}}, "EndTime": 1552274857.244055, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274857.244039}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.4989315968422434, "sum": 0.4989315968422434, "min": 0.4989315968422434}}, "EndTime": 1552274857.24411, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274857.244095}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5090775651787993, "sum": 0.5090775651787993, "min": 0.5090775651787993}}, "EndTime": 1552274857.244159, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274857.244147}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.49894058357890525, "sum": 0.49894058357890525, "min": 0.49894058357890525}}, "EndTime": 1552274857.24421, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274857.244195}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.509083063020179, "sum": 0.509083063020179, "min": 0.509083063020179}}, "EndTime": 1552274857.244266, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274857.244251}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5118569437151578, "sum": 0.5118569437151578, "min": 0.5118569437151578}}, "EndTime": 1552274857.244322, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274857.244306}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5843220768837474, "sum": 0.5843220768837474, "min": 0.5843220768837474}}, "EndTime": 1552274857.244379, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274857.244363}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5118611776073974, "sum": 0.5118611776073974, "min": 0.5118611776073974}}, "EndTime": 1552274857.244433, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274857.244418}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5843062766568745, "sum": 0.5843062766568745, "min": 0.5843062766568745}}, "EndTime": 1552274857.244495, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274857.244479}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1619775651327928, "sum": 1.1619775651327928, "min": 1.1619775651327928}}, "EndTime": 1552274857.244549, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274857.244534}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1656637395350777, "sum": 1.1656637395350777, "min": 1.1656637395350777}}, "EndTime": 1552274857.244601, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274857.244587}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.161973793834897, "sum": 1.161973793834897, "min": 1.161973793834897}}, "EndTime": 1552274857.244664, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274857.244648}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1656635217714548, "sum": 1.1656635217714548, "min": 1.1656635217714548}}, "EndTime": 1552274857.244709, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274857.244695}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1673277857794833, "sum": 1.1673277857794833, "min": 1.1673277857794833}}, "EndTime": 1552274857.244788, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274857.24477}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.292114694336551, "sum": 1.292114694336551, "min": 1.292114694336551}}, "EndTime": 1552274857.244828, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274857.244814}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1673235492035372, "sum": 1.1673235492035372, "min": 1.1673235492035372}}, "EndTime": 1552274857.244879, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274857.244864}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.2920980631766008, "sum": 1.2920980631766008, "min": 1.2920980631766008}}, "EndTime": 1552274857.244934, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274857.244919}
    [0m
    [31m[03/11/2019 03:27:37 INFO 140701971167040] #quality_metric: host=algo-1, epoch=12, train binary_classification_weighted_cross_entropy_objective <loss>=0.341141568093[0m
    [31m[03/11/2019 03:27:37 INFO 140701971167040] #early_stopping_criteria_metric: host=algo-1, epoch=12, criteria=binary_classification_weighted_cross_entropy_objective, value=0.336881507145[0m
    [31m[03/11/2019 03:27:37 INFO 140701971167040] Epoch 12: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:27:37 INFO 140701971167040] #progress_metric: host=algo-1, completed 86 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 2612, "sum": 2612.0, "min": 2612}, "Total Records Seen": {"count": 1, "max": 2603732, "sum": 2603732.0, "min": 2603732}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 15, "sum": 15.0, "min": 15}}, "EndTime": 1552274857.247525, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552274851.804797}
    [0m
    [31m[03/11/2019 03:27:37 INFO 140701971167040] #throughput_metric: host=algo-1, train throughput=36628.4160165 records/second[0m
    [31m[2019-03-11 03:27:37.247] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 14, "duration": 5442, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3403380548678451, "sum": 0.3403380548678451, "min": 0.3403380548678451}}, "EndTime": 1552274863.582591, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274863.582525}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3361356635165574, "sum": 0.3361356635165574, "min": 0.3361356635165574}}, "EndTime": 1552274863.582679, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274863.582661}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3384665114819704, "sum": 0.3384665114819704, "min": 0.3384665114819704}}, "EndTime": 1552274863.58275, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274863.582732}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3372001446671222, "sum": 0.3372001446671222, "min": 0.3372001446671222}}, "EndTime": 1552274863.582818, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274863.5828}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3345176142016847, "sum": 0.3345176142016847, "min": 0.3345176142016847}}, "EndTime": 1552274863.582886, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274863.582868}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5952526120708216, "sum": 0.5952526120708216, "min": 0.5952526120708216}}, "EndTime": 1552274863.582945, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274863.582929}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3348870686286658, "sum": 0.3348870686286658, "min": 0.3348870686286658}}, "EndTime": 1552274863.583001, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274863.582985}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5863257333956771, "sum": 0.5863257333956771, "min": 0.5863257333956771}}, "EndTime": 1552274863.583053, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274863.583039}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34043442167349197, "sum": 0.34043442167349197, "min": 0.34043442167349197}}, "EndTime": 1552274863.58311, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274863.583093}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3407913978231612, "sum": 0.3407913978231612, "min": 0.3407913978231612}}, "EndTime": 1552274863.583167, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274863.583151}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3425602123701393, "sum": 0.3425602123701393, "min": 0.3425602123701393}}, "EndTime": 1552274863.583222, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274863.583207}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.340736872073993, "sum": 0.340736872073993, "min": 0.340736872073993}}, "EndTime": 1552274863.583276, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274863.583261}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3451417777670089, "sum": 0.3451417777670089, "min": 0.3451417777670089}}, "EndTime": 1552274863.583331, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274863.583316}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5988669708098598, "sum": 0.5988669708098598, "min": 0.5988669708098598}}, "EndTime": 1552274863.583396, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274863.583381}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34307314653252835, "sum": 0.34307314653252835, "min": 0.34307314653252835}}, "EndTime": 1552274863.583447, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274863.583433}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6002385617356804, "sum": 0.6002385617356804, "min": 0.6002385617356804}}, "EndTime": 1552274863.583498, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274863.583484}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.49858488594706935, "sum": 0.49858488594706935, "min": 0.49858488594706935}}, "EndTime": 1552274863.583553, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274863.583538}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5089589788542321, "sum": 0.5089589788542321, "min": 0.5089589788542321}}, "EndTime": 1552274863.583608, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274863.583593}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.49859018542419126, "sum": 0.49859018542419126, "min": 0.49859018542419126}}, "EndTime": 1552274863.58366, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274863.583645}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5089629158518422, "sum": 0.5089629158518422, "min": 0.5089629158518422}}, "EndTime": 1552274863.583713, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274863.583698}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5109003112639614, "sum": 0.5109003112639614, "min": 0.5109003112639614}}, "EndTime": 1552274863.583775, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274863.583758}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5825426100151023, "sum": 0.5825426100151023, "min": 0.5825426100151023}}, "EndTime": 1552274863.583839, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274863.583823}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5109039870219015, "sum": 0.5109039870219015, "min": 0.5109039870219015}}, "EndTime": 1552274863.583896, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274863.58388}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5825327654507891, "sum": 0.5825327654507891, "min": 0.5825327654507891}}, "EndTime": 1552274863.583958, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274863.583942}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1617488743767665, "sum": 1.1617488743767665, "min": 1.1617488743767665}}, "EndTime": 1552274863.584014, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274863.583998}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1656883833228644, "sum": 1.1656883833228644, "min": 1.1656883833228644}}, "EndTime": 1552274863.58407, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274863.584055}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1617468632837036, "sum": 1.1617468632837036, "min": 1.1617468632837036}}, "EndTime": 1552274863.584125, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274863.58411}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1656874509265076, "sum": 1.1656874509265076, "min": 1.1656874509265076}}, "EndTime": 1552274863.584181, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274863.584165}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1660039160646984, "sum": 1.1660039160646984, "min": 1.1660039160646984}}, "EndTime": 1552274863.584238, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274863.584223}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.2875859535255625, "sum": 1.2875859535255625, "min": 1.2875859535255625}}, "EndTime": 1552274863.584295, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274863.584279}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1660032005118366, "sum": 1.1660032005118366, "min": 1.1660032005118366}}, "EndTime": 1552274863.584357, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274863.584342}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.2875715526791673, "sum": 1.2875715526791673, "min": 1.2875715526791673}}, "EndTime": 1552274863.584419, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274863.584404}
    [0m
    [31m[03/11/2019 03:27:43 INFO 140701971167040] #quality_metric: host=algo-1, epoch=13, train binary_classification_weighted_cross_entropy_objective <loss>=0.340338054868[0m
    [31m[03/11/2019 03:27:43 INFO 140701971167040] #early_stopping_criteria_metric: host=algo-1, epoch=13, criteria=binary_classification_weighted_cross_entropy_objective, value=0.334517614202[0m
    [31m[03/11/2019 03:27:43 INFO 140701971167040] Epoch 13: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:27:43 INFO 140701971167040] #progress_metric: host=algo-1, completed 93 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 2812, "sum": 2812.0, "min": 2812}, "Total Records Seen": {"count": 1, "max": 2803096, "sum": 2803096.0, "min": 2803096}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 16, "sum": 16.0, "min": 16}}, "EndTime": 1552274863.587013, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552274857.247826}
    [0m
    [31m[03/11/2019 03:27:43 INFO 140701971167040] #throughput_metric: host=algo-1, train throughput=31448.887909 records/second[0m
    [31m[2019-03-11 03:27:43.587] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 15, "duration": 6339, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.33968455390355096, "sum": 0.33968455390355096, "min": 0.33968455390355096}}, "EndTime": 1552274869.898294, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274869.898228}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.33545819808729926, "sum": 0.33545819808729926, "min": 0.33545819808729926}}, "EndTime": 1552274869.898376, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274869.898363}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.337879189074339, "sum": 0.337879189074339, "min": 0.337879189074339}}, "EndTime": 1552274869.898411, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274869.898403}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3364432534452659, "sum": 0.3364432534452659, "min": 0.3364432534452659}}, "EndTime": 1552274869.89845, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274869.898436}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3313742982107191, "sum": 0.3313742982107191, "min": 0.3313742982107191}}, "EndTime": 1552274869.8985, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274869.89849}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5894885120296, "sum": 0.5894885120296, "min": 0.5894885120296}}, "EndTime": 1552274869.898531, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274869.898524}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3312075480935562, "sum": 0.3312075480935562, "min": 0.3312075480935562}}, "EndTime": 1552274869.898558, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274869.898551}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5907595437974786, "sum": 0.5907595437974786, "min": 0.5907595437974786}}, "EndTime": 1552274869.898583, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274869.898577}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3398797196047989, "sum": 0.3398797196047989, "min": 0.3398797196047989}}, "EndTime": 1552274869.89861, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274869.898602}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3401863389135006, "sum": 0.3401863389135006, "min": 0.3401863389135006}}, "EndTime": 1552274869.898661, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274869.898646}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34193371179475257, "sum": 0.34193371179475257, "min": 0.34193371179475257}}, "EndTime": 1552274869.898698, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274869.898689}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3401562111245927, "sum": 0.3401562111245927, "min": 0.3401562111245927}}, "EndTime": 1552274869.898752, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274869.898736}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34104891595888376, "sum": 0.34104891595888376, "min": 0.34104891595888376}}, "EndTime": 1552274869.898817, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274869.898799}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5775458740541084, "sum": 0.5775458740541084, "min": 0.5775458740541084}}, "EndTime": 1552274869.898884, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274869.898867}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34031021612732854, "sum": 0.34031021612732854, "min": 0.34031021612732854}}, "EndTime": 1552274869.898949, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274869.898933}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5386363755422621, "sum": 0.5386363755422621, "min": 0.5386363755422621}}, "EndTime": 1552274869.899024, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274869.899007}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.4982975454665908, "sum": 0.4982975454665908, "min": 0.4982975454665908}}, "EndTime": 1552274869.899081, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274869.899066}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.508840274293219, "sum": 0.508840274293219, "min": 0.508840274293219}}, "EndTime": 1552274869.899143, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274869.899127}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.4983002771732196, "sum": 0.4983002771732196, "min": 0.4983002771732196}}, "EndTime": 1552274869.899206, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274869.899189}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5088429926579922, "sum": 0.5088429926579922, "min": 0.5088429926579922}}, "EndTime": 1552274869.899269, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274869.899253}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.510074233548725, "sum": 0.510074233548725, "min": 0.510074233548725}}, "EndTime": 1552274869.899331, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274869.899315}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5808202184744217, "sum": 0.5808202184744217, "min": 0.5808202184744217}}, "EndTime": 1552274869.899392, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274869.899376}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5100766696642391, "sum": 0.5100766696642391, "min": 0.5100766696642391}}, "EndTime": 1552274869.899453, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274869.899438}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5808142933486095, "sum": 0.5808142933486095, "min": 0.5808142933486095}}, "EndTime": 1552274869.899516, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274869.8995}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1615434646989833, "sum": 1.1615434646989833, "min": 1.1615434646989833}}, "EndTime": 1552274869.899579, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274869.899563}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1657150526190523, "sum": 1.1657150526190523, "min": 1.1657150526190523}}, "EndTime": 1552274869.899632, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274869.899617}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1615423191396435, "sum": 1.1615423191396435, "min": 1.1615423191396435}}, "EndTime": 1552274869.899684, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274869.89967}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1657136181395258, "sum": 1.1657136181395258, "min": 1.1657136181395258}}, "EndTime": 1552274869.899747, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274869.89973}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1653917190321725, "sum": 1.1653917190321725, "min": 1.1653917190321725}}, "EndTime": 1552274869.899803, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274869.899788}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.2832254080460899, "sum": 1.2832254080460899, "min": 1.2832254080460899}}, "EndTime": 1552274869.899864, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274869.899848}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1653905458689935, "sum": 1.1653905458689935, "min": 1.1653905458689935}}, "EndTime": 1552274869.899922, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274869.899906}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.283214558050261, "sum": 1.283214558050261, "min": 1.283214558050261}}, "EndTime": 1552274869.899975, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274869.899961}
    [0m
    [31m[03/11/2019 03:27:49 INFO 140701971167040] #quality_metric: host=algo-1, epoch=14, train binary_classification_weighted_cross_entropy_objective <loss>=0.339684553904[0m
    [31m[03/11/2019 03:27:49 INFO 140701971167040] #early_stopping_criteria_metric: host=algo-1, epoch=14, criteria=binary_classification_weighted_cross_entropy_objective, value=0.331207548094[0m
    [31m[03/11/2019 03:27:49 INFO 140701971167040] Epoch 14: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:27:49 INFO 140701971167040] #progress_metric: host=algo-1, completed 100 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 3012, "sum": 3012.0, "min": 3012}, "Total Records Seen": {"count": 1, "max": 3002460, "sum": 3002460.0, "min": 3002460}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 17, "sum": 17.0, "min": 17}}, "EndTime": 1552274869.902536, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552274863.587289}
    [0m
    [31m[03/11/2019 03:27:49 INFO 140701971167040] #throughput_metric: host=algo-1, train throughput=31568.062898 records/second[0m
    [31m[03/11/2019 03:27:49 WARNING 140701971167040] wait_for_all_workers will not sync workers since the kv store is not running distributed[0m
    [31m[03/11/2019 03:27:49 WARNING 140701971167040] wait_for_all_workers will not sync workers since the kv store is not running distributed[0m
    [31m[2019-03-11 03:27:49.902] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 16, "duration": 6315, "num_examples": 200}[0m
    [31m[2019-03-11 03:27:49.908] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 17, "duration": 5, "num_examples": 1}[0m
    [31m[2019-03-11 03:27:50.611] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 18, "duration": 700, "num_examples": 200}[0m
    [31m[03/11/2019 03:27:51 INFO 140701971167040] #train_score (algo-1) : ('binary_classification_weighted_cross_entropy_objective', 0.32092717971047102)[0m
    [31m[03/11/2019 03:27:51 INFO 140701971167040] #train_score (algo-1) : ('binary_classification_accuracy', 0.98819245199735162)[0m
    [31m[03/11/2019 03:27:51 INFO 140701971167040] #train_score (algo-1) : ('binary_f_1.000', 0.2116543871399866)[0m
    [31m[03/11/2019 03:27:51 INFO 140701971167040] #train_score (algo-1) : ('precision', 0.11992409867172675)[0m
    [31m[03/11/2019 03:27:51 INFO 140701971167040] #train_score (algo-1) : ('recall', 0.9002849002849003)[0m
    [31m[03/11/2019 03:27:51 INFO 140701971167040] #quality_metric: host=algo-1, train binary_classification_weighted_cross_entropy_objective <loss>=0.32092717971[0m
    [31m[03/11/2019 03:27:51 INFO 140701971167040] #quality_metric: host=algo-1, train binary_classification_accuracy <score>=0.988192451997[0m
    [31m[03/11/2019 03:27:51 INFO 140701971167040] #quality_metric: host=algo-1, train binary_f_1.000 <score>=0.21165438714[0m
    [31m[03/11/2019 03:27:51 INFO 140701971167040] #quality_metric: host=algo-1, train precision <score>=0.119924098672[0m
    [31m[03/11/2019 03:27:51 INFO 140701971167040] #quality_metric: host=algo-1, train recall <score>=0.900284900285[0m
    [31m[03/11/2019 03:27:51 INFO 140701971167040] Best model found for hyperparameters: {"lr_scheduler_step": 10, "wd": 0.0001, "optimizer": "adam", "lr_scheduler_factor": 0.99, "l1": 0.0, "learning_rate": 0.1, "lr_scheduler_minimum_lr": 0.0001}[0m
    [31m[03/11/2019 03:27:51 INFO 140701971167040] Saved checkpoint to "/tmp/tmpm1mo5P/mx-mod-0000.params"[0m
    [31m[03/11/2019 03:27:51 INFO 140701971167040] Test data is not provided.[0m
    [31m[2019-03-11 03:27:51.257] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 19, "duration": 645, "num_examples": 200}[0m
    [31m[2019-03-11 03:27:51.257] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "duration": 96204, "num_epochs": 20, "num_examples": 3413}[0m
    [31m#metrics {"Metrics": {"totaltime": {"count": 1, "max": 96468.39189529419, "sum": 96468.39189529419, "min": 96468.39189529419}, "finalize.time": {"count": 1, "max": 1342.383861541748, "sum": 1342.383861541748, "min": 1342.383861541748}, "initialize.time": {"count": 1, "max": 248.78311157226562, "sum": 248.78311157226562, "min": 248.78311157226562}, "check_early_stopping.time": {"count": 15, "max": 0.9000301361083984, "sum": 12.03298568725586, "min": 0.7500648498535156}, "setuptime": {"count": 1, "max": 14.410972595214844, "sum": 14.410972595214844, "min": 14.410972595214844}, "update.time": {"count": 15, "max": 6929.7990798950195, "sum": 94765.53416252136, "min": 5442.5811767578125}, "epochs": {"count": 1, "max": 15, "sum": 15.0, "min": 15}}, "EndTime": 1552274871.258231, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1552274774.875197}
    [0m
    
    2019-03-11 03:27:57 Uploading - Uploading generated training model
    2019-03-11 03:27:57 Completed - Training job completed
    Billable seconds: 117
    CPU times: user 597 ms, sys: 61.2 ms, total: 659 ms
    Wall time: 4min 43s


### EXERCISE: Deploy and evaluate the balanced estimator

Deploy the balanced predictor and evaluate it. Do the results match with your expectations?


```python
%%time 
# deploy and create a predictor
balanced_predictor = linear_balanced.deploy(initial_instance_count=1, instance_type='ml.t2.medium')
```

    INFO:sagemaker:Creating model with name: linear-learner-2019-03-11-03-28-23-037
    INFO:sagemaker:Creating endpoint with name linear-learner-2019-03-11-03-23-39-913


    ---------------------------------------------------------------------------------------------------!CPU times: user 494 ms, sys: 31 ms, total: 525 ms
    Wall time: 8min 21s



```python
print('Metrics for balanced, LinearLearner.\n')

# get metrics for balanced predictor
metrics = evaluate(balanced_predictor, 
                   test_features.astype('float32'), 
                   test_labels, 
                   verbose=True)
```

    Metrics for balanced, LinearLearner.
    
    prediction (col)    0.0   1.0
    actual (row)                 
    0.0               84277  1025
    1.0                  12   129
    
    Recall:     0.915
    Precision:  0.112
    Accuracy:   0.988
    


## Delete the endpoint 

When you're done evaluating a model, you should delete the endpoint.


```python
# delete the predictor endpoint 
delete_endpoint(balanced_predictor)
```

    Deleted linear-learner-2019-03-11-03-23-39-913


A note on metric variability: 

The above model is tuned for the best possible precision with recall fixed at about 90%. The recall is fixed at 90% during training, but may vary when we apply our trained model to a test set of data.

---
## Model Design

Now that you've seen how to tune and balance a LinearLearner. Create, train and deploy your own model. This exercise is meant to be more open-ended, so that you get practice with the steps involved in designing a model and deploying it.

### EXERCISE: Train and deploy a LinearLearner with appropriate hyperparameters, according to the given scenario

**Scenario:**
* A bank has asked you to build a model that optimizes for a good user experience; users should only ever have up to about 15% of their valid transactions flagged as fraudulent.

This requires that you make a design decision: Given the above scenario, what metric (and value) should you aim for during training?

You may assume that performance on a training set will be within about 5-10% of the performance on a test set. For example, if you get 80% on a training set, you can assume that you'll get between about 70-90% accuracy on a test set.

Your final model should account for class imbalance and be appropriately tuned. 

If we're allowed about 15/100 incorrectly classified valid transactions (false positives), then I can calculate an approximate value for the precision that I want as: 85/(85+15) = 85%. I'll aim for about 5% higher during training to ensure that I get closer to 80-85% precision on the test data.


```python
%%time
# instantiate and train a LinearLearner

# include params for tuning for higher precision
# *and* account for class imbalance in training data
linear_precision = LinearLearner(role=role,
                                train_instance_count=1, 
                                train_instance_type='ml.c4.xlarge',
                                predictor_type='binary_classifier',
                                output_path=output_path,
                                sagemaker_session=sagemaker_session,
                                epochs=15,
                                binary_classifier_model_selection_criteria='recall_at_target_precision',
                                target_precision=0.9,
                                positive_example_weight_mult='balanced')


# train the estimator on formatted training data
linear_precision.fit(formatted_train_data)
```

    INFO:sagemaker:Creating training-job with name: linear-learner-2019-03-11-03-36-56-524


    2019-03-11 03:36:56 Starting - Starting the training job...
    2019-03-11 03:37:00 Starting - Launching requested ML instances......
    2019-03-11 03:38:01 Starting - Preparing the instances for training.........
    2019-03-11 03:39:44 Downloading - Downloading input data
    2019-03-11 03:39:44 Training - Training image download completed. Training in progress.
    [31mDocker entrypoint called with argument(s): train[0m
    [31m[03/11/2019 03:39:46 INFO 140575172896576] Reading default configuration from /opt/amazon/lib/python2.7/site-packages/algorithm/default-input.json: {u'loss_insensitivity': u'0.01', u'epochs': u'15', u'init_bias': u'0.0', u'lr_scheduler_factor': u'auto', u'num_calibration_samples': u'10000000', u'accuracy_top_k': u'3', u'_num_kv_servers': u'auto', u'use_bias': u'true', u'num_point_for_scaler': u'10000', u'_log_level': u'info', u'quantile': u'0.5', u'bias_lr_mult': u'auto', u'lr_scheduler_step': u'auto', u'init_method': u'uniform', u'init_sigma': u'0.01', u'lr_scheduler_minimum_lr': u'auto', u'target_recall': u'0.8', u'num_models': u'auto', u'early_stopping_patience': u'3', u'momentum': u'auto', u'unbias_label': u'auto', u'wd': u'auto', u'optimizer': u'auto', u'_tuning_objective_metric': u'', u'early_stopping_tolerance': u'0.001', u'learning_rate': u'auto', u'_kvstore': u'auto', u'normalize_data': u'true', u'binary_classifier_model_selection_criteria': u'accuracy', u'use_lr_scheduler': u'true', u'target_precision': u'0.8', u'unbias_data': u'auto', u'init_scale': u'0.07', u'bias_wd_mult': u'auto', u'f_beta': u'1.0', u'mini_batch_size': u'1000', u'huber_delta': u'1.0', u'num_classes': u'1', u'beta_1': u'auto', u'loss': u'auto', u'beta_2': u'auto', u'_enable_profiler': u'false', u'normalize_label': u'auto', u'_num_gpus': u'auto', u'balance_multiclass_weights': u'false', u'positive_example_weight_mult': u'1.0', u'l1': u'auto', u'margin': u'1.0'}[0m
    [31m[03/11/2019 03:39:46 INFO 140575172896576] Reading provided configuration from /opt/ml/input/config/hyperparameters.json: {u'predictor_type': u'binary_classifier', u'feature_dim': u'30', u'target_precision': u'0.9', u'binary_classifier_model_selection_criteria': u'recall_at_target_precision', u'epochs': u'15', u'positive_example_weight_mult': u'balanced', u'mini_batch_size': u'1000'}[0m
    [31m[03/11/2019 03:39:46 INFO 140575172896576] Final configuration: {u'loss_insensitivity': u'0.01', u'epochs': u'15', u'feature_dim': u'30', u'init_bias': u'0.0', u'lr_scheduler_factor': u'auto', u'num_calibration_samples': u'10000000', u'accuracy_top_k': u'3', u'_num_kv_servers': u'auto', u'use_bias': u'true', u'num_point_for_scaler': u'10000', u'_log_level': u'info', u'quantile': u'0.5', u'bias_lr_mult': u'auto', u'lr_scheduler_step': u'auto', u'init_method': u'uniform', u'init_sigma': u'0.01', u'lr_scheduler_minimum_lr': u'auto', u'target_recall': u'0.8', u'num_models': u'auto', u'early_stopping_patience': u'3', u'momentum': u'auto', u'unbias_label': u'auto', u'wd': u'auto', u'optimizer': u'auto', u'_tuning_objective_metric': u'', u'early_stopping_tolerance': u'0.001', u'learning_rate': u'auto', u'_kvstore': u'auto', u'normalize_data': u'true', u'binary_classifier_model_selection_criteria': u'recall_at_target_precision', u'use_lr_scheduler': u'true', u'target_precision': u'0.9', u'unbias_data': u'auto', u'init_scale': u'0.07', u'bias_wd_mult': u'auto', u'f_beta': u'1.0', u'mini_batch_size': u'1000', u'huber_delta': u'1.0', u'num_classes': u'1', u'predictor_type': u'binary_classifier', u'beta_1': u'auto', u'loss': u'auto', u'beta_2': u'auto', u'_enable_profiler': u'false', u'normalize_label': u'auto', u'_num_gpus': u'auto', u'balance_multiclass_weights': u'false', u'positive_example_weight_mult': u'balanced', u'l1': u'auto', u'margin': u'1.0'}[0m
    [31m[03/11/2019 03:39:46 WARNING 140575172896576] Loggers have already been setup.[0m
    [31mProcess 1 is a worker.[0m
    [31m[03/11/2019 03:39:46 INFO 140575172896576] Using default worker.[0m
    [31m[2019-03-11 03:39:46.271] [tensorio] [info] batch={"data_pipeline": "/opt/ml/input/data/train", "num_examples": 1000, "features": [{"name": "label_values", "shape": [1], "storage_type": "dense"}, {"name": "values", "shape": [30], "storage_type": "dense"}]}[0m
    [31m[2019-03-11 03:39:46.296] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 0, "duration": 25, "num_examples": 1}[0m
    [31m[03/11/2019 03:39:46 INFO 140575172896576] Create Store: local[0m
    [31m[2019-03-11 03:39:46.356] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 1, "duration": 58, "num_examples": 11}[0m
    [31m[03/11/2019 03:39:46 INFO 140575172896576] Scaler algorithm parameters
     <algorithm.scaler.ScalerAlgorithmStable object at 0x7fd9e50faa50>[0m
    [31m[03/11/2019 03:39:46 INFO 140575172896576] Scaling model computed with parameters:
     {'stdev_weight': [0m
    [31m[  4.75497891e+04   2.01225400e+00   1.72936726e+00   1.48752689e+00
       1.41830683e+00   1.42959750e+00   1.34760964e+00   1.27067423e+00
       1.24293745e+00   1.09265101e+00   1.05321789e+00   1.01260686e+00
       9.87991810e-01   1.00782645e+00   9.47202206e-01   9.02963459e-01
       8.68877888e-01   8.27179432e-01   8.36477458e-01   8.07050884e-01
       8.00110519e-01   7.55493522e-01   7.21427202e-01   6.25614405e-01
       6.10876381e-01   5.16283095e-01   4.88118291e-01   4.35698181e-01
       3.69419903e-01   2.47155548e+02][0m
    [31m<NDArray 30 @cpu(0)>, 'stdev_label': None, 'mean_label': None, 'mean_weight': [0m
    [31m[  9.44802812e+04  -1.04726264e-02  -1.43008800e-02   1.28451567e-02
       1.87512934e-02  -2.48281248e-02   5.86199807e-03  -7.13069551e-03
      -7.39883492e-03   1.20382467e-02   6.10911567e-03  -3.16866231e-03
       8.64854374e-04   2.46435311e-03   1.56665407e-02   1.12619074e-02
      -4.91584092e-03  -1.56447978e-03   2.45723873e-03   2.82235094e-04
      -3.25949211e-03   6.57527940e-03   3.11945518e-03   6.22356636e-03
      -6.13171898e-04  -3.88089707e-03   1.16021503e-02  -3.21021304e-03
      -5.27510792e-03   8.94287567e+01][0m
    [31m<NDArray 30 @cpu(0)>}[0m
    [31m[03/11/2019 03:39:46 INFO 140575172896576] nvidia-smi took: 0.0251851081848 secs to identify 0 gpus[0m
    [31m[03/11/2019 03:39:46 INFO 140575172896576] Number of GPUs being used: 0[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 11, "sum": 11.0, "min": 11}, "Number of Batches Since Last Reset": {"count": 1, "max": 0, "sum": 0.0, "min": 0}, "Number of Records Since Last Reset": {"count": 1, "max": 0, "sum": 0.0, "min": 0}, "Total Batches Seen": {"count": 1, "max": 12, "sum": 12.0, "min": 12}, "Total Records Seen": {"count": 1, "max": 12000, "sum": 12000.0, "min": 12000}, "Max Records Seen Between Resets": {"count": 1, "max": 11000, "sum": 11000.0, "min": 11000}, "Reset Count": {"count": 1, "max": 2, "sum": 2.0, "min": 2}}, "EndTime": 1552275586.511961, "Dimensions": {"Host": "algo-1", "Meta": "init_train_data_iter", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1552275586.511925}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6594284648799417, "sum": 0.6594284648799417, "min": 0.6594284648799417}}, "EndTime": 1552275592.757046, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552275592.756981}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5641238785940199, "sum": 0.5641238785940199, "min": 0.5641238785940199}}, "EndTime": 1552275592.757135, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552275592.757115}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5921783411994053, "sum": 0.5921783411994053, "min": 0.5921783411994053}}, "EndTime": 1552275592.757192, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552275592.757176}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6301838542228967, "sum": 0.6301838542228967, "min": 0.6301838542228967}}, "EndTime": 1552275592.757273, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552275592.757257}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6843703128297125, "sum": 0.6843703128297125, "min": 0.6843703128297125}}, "EndTime": 1552275592.757322, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552275592.757309}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6846481458385986, "sum": 0.6846481458385986, "min": 0.6846481458385986}}, "EndTime": 1552275592.757371, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552275592.757357}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6839004336313985, "sum": 0.6839004336313985, "min": 0.6839004336313985}}, "EndTime": 1552275592.757426, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552275592.757411}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.7122602933567374, "sum": 0.7122602933567374, "min": 0.7122602933567374}}, "EndTime": 1552275592.757483, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552275592.757466}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5813303722592454, "sum": 0.5813303722592454, "min": 0.5813303722592454}}, "EndTime": 1552275592.757538, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552275592.757523}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6624351184020689, "sum": 0.6624351184020689, "min": 0.6624351184020689}}, "EndTime": 1552275592.757594, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552275592.757578}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6498399631366059, "sum": 0.6498399631366059, "min": 0.6498399631366059}}, "EndTime": 1552275592.757648, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552275592.757633}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6519365340189718, "sum": 0.6519365340189718, "min": 0.6519365340189718}}, "EndTime": 1552275592.757703, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552275592.757688}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6812790213350075, "sum": 0.6812790213350075, "min": 0.6812790213350075}}, "EndTime": 1552275592.757756, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552275592.757741}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6948909245591667, "sum": 0.6948909245591667, "min": 0.6948909245591667}}, "EndTime": 1552275592.757822, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552275592.757797}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6716007738161327, "sum": 0.6716007738161327, "min": 0.6716007738161327}}, "EndTime": 1552275592.757882, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552275592.757867}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6938190375476626, "sum": 0.6938190375476626, "min": 0.6938190375476626}}, "EndTime": 1552275592.75794, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552275592.757925}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6940059118989724, "sum": 0.6940059118989724, "min": 0.6940059118989724}}, "EndTime": 1552275592.758, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552275592.757984}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5978640082277844, "sum": 0.5978640082277844, "min": 0.5978640082277844}}, "EndTime": 1552275592.758063, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552275592.758048}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6243885920538974, "sum": 0.6243885920538974, "min": 0.6243885920538974}}, "EndTime": 1552275592.758121, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552275592.758105}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6476971526026126, "sum": 0.6476971526026126, "min": 0.6476971526026126}}, "EndTime": 1552275592.758175, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552275592.758159}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.63529909008352, "sum": 0.63529909008352, "min": 0.63529909008352}}, "EndTime": 1552275592.75823, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552275592.758214}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6430660022563072, "sum": 0.6430660022563072, "min": 0.6430660022563072}}, "EndTime": 1552275592.758286, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552275592.758271}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6422417014999007, "sum": 0.6422417014999007, "min": 0.6422417014999007}}, "EndTime": 1552275592.75834, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552275592.758325}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6510663895918496, "sum": 0.6510663895918496, "min": 0.6510663895918496}}, "EndTime": 1552275592.758382, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552275592.75837}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1656365715295227, "sum": 1.1656365715295227, "min": 1.1656365715295227}}, "EndTime": 1552275592.758432, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552275592.758417}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1654303487653108, "sum": 1.1654303487653108, "min": 1.1654303487653108}}, "EndTime": 1552275592.75851, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552275592.758498}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1970537133911747, "sum": 1.1970537133911747, "min": 1.1970537133911747}}, "EndTime": 1552275592.758561, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552275592.758547}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.171867919921875, "sum": 1.171867919921875, "min": 1.171867919921875}}, "EndTime": 1552275592.758615, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552275592.7586}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.3268262262392283, "sum": 1.3268262262392283, "min": 1.3268262262392283}}, "EndTime": 1552275592.758666, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552275592.758651}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.3632041089235238, "sum": 1.3632041089235238, "min": 1.3632041089235238}}, "EndTime": 1552275592.758726, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552275592.75871}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.332283373348677, "sum": 1.332283373348677, "min": 1.332283373348677}}, "EndTime": 1552275592.758783, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552275592.758768}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.3859594543303677, "sum": 1.3859594543303677, "min": 1.3859594543303677}}, "EndTime": 1552275592.758841, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552275592.758826}
    [0m
    [31m[03/11/2019 03:39:52 INFO 140575172896576] #quality_metric: host=algo-1, epoch=0, train binary_classification_weighted_cross_entropy_objective <loss>=0.65942846488[0m
    [31m[03/11/2019 03:39:52 INFO 140575172896576] #early_stopping_criteria_metric: host=algo-1, epoch=0, criteria=binary_classification_weighted_cross_entropy_objective, value=0.564123878594[0m
    [31m[03/11/2019 03:39:52 INFO 140575172896576] Epoch 0: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:39:52 INFO 140575172896576] #progress_metric: host=algo-1, completed 6 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 212, "sum": 212.0, "min": 212}, "Total Records Seen": {"count": 1, "max": 211364, "sum": 211364.0, "min": 211364}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 3, "sum": 3.0, "min": 3}}, "EndTime": 1552275592.763853, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1552275586.512148}
    [0m
    [31m[03/11/2019 03:39:52 INFO 140575172896576] #throughput_metric: host=algo-1, train throughput=31888.913359 records/second[0m
    [31m[2019-03-11 03:39:52.764] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 2, "duration": 6251, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.4400108514526981, "sum": 0.4400108514526981, "min": 0.4400108514526981}}, "EndTime": 1552275599.311408, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552275599.311348}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.41052518957703554, "sum": 0.41052518957703554, "min": 0.41052518957703554}}, "EndTime": 1552275599.311493, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552275599.311475}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.4252546185057367, "sum": 0.4252546185057367, "min": 0.4252546185057367}}, "EndTime": 1552275599.311549, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552275599.311534}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.42712178238911847, "sum": 0.42712178238911847, "min": 0.42712178238911847}}, "EndTime": 1552275599.311602, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552275599.311588}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5716547131178966, "sum": 0.5716547131178966, "min": 0.5716547131178966}}, "EndTime": 1552275599.311651, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552275599.311638}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.670325926584215, "sum": 0.670325926584215, "min": 0.670325926584215}}, "EndTime": 1552275599.311697, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552275599.311684}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5771199898647903, "sum": 0.5771199898647903, "min": 0.5771199898647903}}, "EndTime": 1552275599.311751, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552275599.311736}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6394495888714814, "sum": 0.6394495888714814, "min": 0.6394495888714814}}, "EndTime": 1552275599.311831, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552275599.311814}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.4237853919561185, "sum": 0.4237853919561185, "min": 0.4237853919561185}}, "EndTime": 1552275599.311896, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552275599.311879}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.43772098046690977, "sum": 0.43772098046690977, "min": 0.43772098046690977}}, "EndTime": 1552275599.311954, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552275599.311938}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.4373177432726376, "sum": 0.4373177432726376, "min": 0.4373177432726376}}, "EndTime": 1552275599.31201, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552275599.311995}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.4355643713102868, "sum": 0.4355643713102868, "min": 0.4355643713102868}}, "EndTime": 1552275599.312073, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552275599.312057}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5609734526016006, "sum": 0.5609734526016006, "min": 0.5609734526016006}}, "EndTime": 1552275599.312145, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552275599.31213}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6396466855763191, "sum": 0.6396466855763191, "min": 0.6396466855763191}}, "EndTime": 1552275599.312199, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552275599.312185}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5679037587534842, "sum": 0.5679037587534842, "min": 0.5679037587534842}}, "EndTime": 1552275599.312251, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552275599.312237}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6381025119378938, "sum": 0.6381025119378938, "min": 0.6381025119378938}}, "EndTime": 1552275599.312302, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552275599.312288}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5228390950725306, "sum": 0.5228390950725306, "min": 0.5228390950725306}}, "EndTime": 1552275599.312354, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552275599.312339}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5129850728403983, "sum": 0.5129850728403983, "min": 0.5129850728403983}}, "EndTime": 1552275599.312404, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552275599.31239}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5143203467747673, "sum": 0.5143203467747673, "min": 0.5143203467747673}}, "EndTime": 1552275599.312456, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552275599.312442}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5161900682305571, "sum": 0.5161900682305571, "min": 0.5161900682305571}}, "EndTime": 1552275599.31251, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552275599.312495}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5769340392452987, "sum": 0.5769340392452987, "min": 0.5769340392452987}}, "EndTime": 1552275599.312563, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552275599.312549}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6002559302056855, "sum": 0.6002559302056855, "min": 0.6002559302056855}}, "EndTime": 1552275599.312613, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552275599.312599}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5767329220028978, "sum": 0.5767329220028978, "min": 0.5767329220028978}}, "EndTime": 1552275599.312667, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552275599.312653}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6016254009074302, "sum": 0.6016254009074302, "min": 0.6016254009074302}}, "EndTime": 1552275599.312727, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552275599.312714}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1656406513770021, "sum": 1.1656406513770021, "min": 1.1656406513770021}}, "EndTime": 1552275599.312785, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552275599.312768}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1657563387616794, "sum": 1.1657563387616794, "min": 1.1657563387616794}}, "EndTime": 1552275599.312845, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552275599.31283}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1665071908001923, "sum": 1.1665071908001923, "min": 1.1665071908001923}}, "EndTime": 1552275599.312906, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552275599.31289}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1657475589004593, "sum": 1.1657475589004593, "min": 1.1657475589004593}}, "EndTime": 1552275599.312975, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552275599.312958}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.268502646748145, "sum": 1.268502646748145, "min": 1.268502646748145}}, "EndTime": 1552275599.31303, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552275599.313014}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.3813459926586056, "sum": 1.3813459926586056, "min": 1.3813459926586056}}, "EndTime": 1552275599.313093, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552275599.313076}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.2631105847382664, "sum": 1.2631105847382664, "min": 1.2631105847382664}}, "EndTime": 1552275599.313156, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552275599.313139}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.3678211852413924, "sum": 1.3678211852413924, "min": 1.3678211852413924}}, "EndTime": 1552275599.313212, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552275599.313197}
    [0m
    [31m[03/11/2019 03:39:59 INFO 140575172896576] #quality_metric: host=algo-1, epoch=1, train binary_classification_weighted_cross_entropy_objective <loss>=0.440010851453[0m
    [31m[03/11/2019 03:39:59 INFO 140575172896576] #early_stopping_criteria_metric: host=algo-1, epoch=1, criteria=binary_classification_weighted_cross_entropy_objective, value=0.410525189577[0m
    [31m[03/11/2019 03:39:59 INFO 140575172896576] Epoch 1: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:39:59 INFO 140575172896576] #progress_metric: host=algo-1, completed 13 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 412, "sum": 412.0, "min": 412}, "Total Records Seen": {"count": 1, "max": 410728, "sum": 410728.0, "min": 410728}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 4, "sum": 4.0, "min": 4}}, "EndTime": 1552275599.315903, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1552275592.764153}
    [0m
    [31m[03/11/2019 03:39:59 INFO 140575172896576] #throughput_metric: host=algo-1, train throughput=30428.5841004 records/second[0m
    [31m[2019-03-11 03:39:59.316] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 3, "duration": 6551, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.39636924766655546, "sum": 0.39636924766655546, "min": 0.39636924766655546}}, "EndTime": 1552275605.831702, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552275605.831643}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.376585278074945, "sum": 0.376585278074945, "min": 0.376585278074945}}, "EndTime": 1552275605.831795, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552275605.831763}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.38781511101650834, "sum": 0.38781511101650834, "min": 0.38781511101650834}}, "EndTime": 1552275605.831855, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552275605.831839}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.38641533173987613, "sum": 0.38641533173987613, "min": 0.38641533173987613}}, "EndTime": 1552275605.831916, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552275605.831901}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5263680824394801, "sum": 0.5263680824394801, "min": 0.5263680824394801}}, "EndTime": 1552275605.831976, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552275605.83196}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6410529355571498, "sum": 0.6410529355571498, "min": 0.6410529355571498}}, "EndTime": 1552275605.832032, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552275605.832016}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.527668864609608, "sum": 0.527668864609608, "min": 0.527668864609608}}, "EndTime": 1552275605.832087, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552275605.832072}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6855561453852821, "sum": 0.6855561453852821, "min": 0.6855561453852821}}, "EndTime": 1552275605.832138, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552275605.832124}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.38759695419234846, "sum": 0.38759695419234846, "min": 0.38759695419234846}}, "EndTime": 1552275605.832187, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552275605.832173}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.39293417833797895, "sum": 0.39293417833797895, "min": 0.39293417833797895}}, "EndTime": 1552275605.832236, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552275605.832223}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3962465733667115, "sum": 0.3962465733667115, "min": 0.3962465733667115}}, "EndTime": 1552275605.832288, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552275605.832273}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.39143112677185976, "sum": 0.39143112677185976, "min": 0.39143112677185976}}, "EndTime": 1552275605.83234, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552275605.832325}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5372402820491311, "sum": 0.5372402820491311, "min": 0.5372402820491311}}, "EndTime": 1552275605.8324, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552275605.832386}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6481121471347522, "sum": 0.6481121471347522, "min": 0.6481121471347522}}, "EndTime": 1552275605.832451, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552275605.832437}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5270041325631453, "sum": 0.5270041325631453, "min": 0.5270041325631453}}, "EndTime": 1552275605.832501, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552275605.832487}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6470744394657001, "sum": 0.6470744394657001, "min": 0.6470744394657001}}, "EndTime": 1552275605.832551, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552275605.832537}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5103085941333867, "sum": 0.5103085941333867, "min": 0.5103085941333867}}, "EndTime": 1552275605.832581, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552275605.832574}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5106808740050349, "sum": 0.5106808740050349, "min": 0.5106808740050349}}, "EndTime": 1552275605.832607, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552275605.8326}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5085814014990725, "sum": 0.5085814014990725, "min": 0.5085814014990725}}, "EndTime": 1552275605.832632, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552275605.832626}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5110635287030857, "sum": 0.5110635287030857, "min": 0.5110635287030857}}, "EndTime": 1552275605.832665, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552275605.832658}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.565871770734164, "sum": 0.565871770734164, "min": 0.565871770734164}}, "EndTime": 1552275605.832692, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552275605.832685}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5999275546625031, "sum": 0.5999275546625031, "min": 0.5999275546625031}}, "EndTime": 1552275605.832717, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552275605.83271}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.566059082146266, "sum": 0.566059082146266, "min": 0.566059082146266}}, "EndTime": 1552275605.832742, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552275605.832736}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5999852474921912, "sum": 0.5999852474921912, "min": 0.5999852474921912}}, "EndTime": 1552275605.832767, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552275605.832761}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.166252090530779, "sum": 1.166252090530779, "min": 1.166252090530779}}, "EndTime": 1552275605.832792, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552275605.832786}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1656566413611025, "sum": 1.1656566413611025, "min": 1.1656566413611025}}, "EndTime": 1552275605.832835, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552275605.832821}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1664664858717415, "sum": 1.1664664858717415, "min": 1.1664664858717415}}, "EndTime": 1552275605.832874, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552275605.832861}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.165680956969908, "sum": 1.165680956969908, "min": 1.165680956969908}}, "EndTime": 1552275605.832934, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552275605.832919}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.2412207515083964, "sum": 1.2412207515083964, "min": 1.2412207515083964}}, "EndTime": 1552275605.832974, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552275605.832964}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.360423379332576, "sum": 1.360423379332576, "min": 1.360423379332576}}, "EndTime": 1552275605.833027, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552275605.833016}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.2389932212637897, "sum": 1.2389932212637897, "min": 1.2389932212637897}}, "EndTime": 1552275605.833075, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552275605.833065}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.356852158666256, "sum": 1.356852158666256, "min": 1.356852158666256}}, "EndTime": 1552275605.833111, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552275605.833103}
    [0m
    [31m[03/11/2019 03:40:05 INFO 140575172896576] #quality_metric: host=algo-1, epoch=2, train binary_classification_weighted_cross_entropy_objective <loss>=0.396369247667[0m
    [31m[03/11/2019 03:40:05 INFO 140575172896576] #early_stopping_criteria_metric: host=algo-1, epoch=2, criteria=binary_classification_weighted_cross_entropy_objective, value=0.376585278075[0m
    [31m[03/11/2019 03:40:05 INFO 140575172896576] Epoch 2: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:40:05 INFO 140575172896576] #progress_metric: host=algo-1, completed 20 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 612, "sum": 612.0, "min": 612}, "Total Records Seen": {"count": 1, "max": 610092, "sum": 610092.0, "min": 610092}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 5, "sum": 5.0, "min": 5}}, "EndTime": 1552275605.836091, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1552275599.31618}
    [0m
    [31m[03/11/2019 03:40:05 INFO 140575172896576] #throughput_metric: host=algo-1, train throughput=30577.0748427 records/second[0m
    [31m[2019-03-11 03:40:05.836] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 4, "duration": 6519, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3764211182714108, "sum": 0.3764211182714108, "min": 0.3764211182714108}}, "EndTime": 1552275612.137956, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552275612.1379}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3615751494211168, "sum": 0.3615751494211168, "min": 0.3615751494211168}}, "EndTime": 1552275612.138029, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552275612.138016}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3704841190510659, "sum": 0.3704841190510659, "min": 0.3704841190510659}}, "EndTime": 1552275612.138063, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552275612.138055}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3682524187097597, "sum": 0.3682524187097597, "min": 0.3682524187097597}}, "EndTime": 1552275612.138096, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552275612.138088}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.497972018179582, "sum": 0.497972018179582, "min": 0.497972018179582}}, "EndTime": 1552275612.138124, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552275612.138117}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.658230786692557, "sum": 0.658230786692557, "min": 0.658230786692557}}, "EndTime": 1552275612.138151, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552275612.138144}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.4991150675634643, "sum": 0.4991150675634643, "min": 0.4991150675634643}}, "EndTime": 1552275612.138177, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552275612.13817}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6169741138285728, "sum": 0.6169741138285728, "min": 0.6169741138285728}}, "EndTime": 1552275612.138203, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552275612.138196}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3709429301352956, "sum": 0.3709429301352956, "min": 0.3709429301352956}}, "EndTime": 1552275612.138228, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552275612.138221}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3725585856222028, "sum": 0.3725585856222028, "min": 0.3725585856222028}}, "EndTime": 1552275612.138253, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552275612.138247}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.37727097776786767, "sum": 0.37727097776786767, "min": 0.37727097776786767}}, "EndTime": 1552275612.138279, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552275612.138272}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.37159369559743294, "sum": 0.37159369559743294, "min": 0.37159369559743294}}, "EndTime": 1552275612.138305, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552275612.138298}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5094950629766263, "sum": 0.5094950629766263, "min": 0.5094950629766263}}, "EndTime": 1552275612.13833, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552275612.138324}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6470716767239211, "sum": 0.6470716767239211, "min": 0.6470716767239211}}, "EndTime": 1552275612.138356, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552275612.138349}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5039542786487982, "sum": 0.5039542786487982, "min": 0.5039542786487982}}, "EndTime": 1552275612.138385, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552275612.138378}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6462183145877703, "sum": 0.6462183145877703, "min": 0.6462183145877703}}, "EndTime": 1552275612.138413, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552275612.138406}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5069897746943949, "sum": 0.5069897746943949, "min": 0.5069897746943949}}, "EndTime": 1552275612.138439, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552275612.138432}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5102628221368071, "sum": 0.5102628221368071, "min": 0.5102628221368071}}, "EndTime": 1552275612.138464, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552275612.138457}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5066128204192348, "sum": 0.5066128204192348, "min": 0.5066128204192348}}, "EndTime": 1552275612.138489, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552275612.138482}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5103834318994876, "sum": 0.5103834318994876, "min": 0.5103834318994876}}, "EndTime": 1552275612.138514, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552275612.138507}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5548706218393604, "sum": 0.5548706218393604, "min": 0.5548706218393604}}, "EndTime": 1552275612.138539, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552275612.138532}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5986631354902258, "sum": 0.5986631354902258, "min": 0.5986631354902258}}, "EndTime": 1552275612.138564, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552275612.138557}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.554863652444964, "sum": 0.554863652444964, "min": 0.554863652444964}}, "EndTime": 1552275612.138588, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552275612.138582}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5985999432664421, "sum": 0.5985999432664421, "min": 0.5985999432664421}}, "EndTime": 1552275612.138613, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552275612.138607}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.166377805163513, "sum": 1.166377805163513, "min": 1.166377805163513}}, "EndTime": 1552275612.138638, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552275612.138632}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1656277021091788, "sum": 1.1656277021091788, "min": 1.1656277021091788}}, "EndTime": 1552275612.138663, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552275612.138657}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1663435764025205, "sum": 1.1663435764025205, "min": 1.1663435764025205}}, "EndTime": 1552275612.138689, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552275612.138683}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1656497630977152, "sum": 1.1656497630977152, "min": 1.1656497630977152}}, "EndTime": 1552275612.138715, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552275612.138708}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.226151385896769, "sum": 1.226151385896769, "min": 1.226151385896769}}, "EndTime": 1552275612.13875, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552275612.138744}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.3499431781097873, "sum": 1.3499431781097873, "min": 1.3499431781097873}}, "EndTime": 1552275612.138775, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552275612.138768}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.2239721780901578, "sum": 1.2239721780901578, "min": 1.2239721780901578}}, "EndTime": 1552275612.138799, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552275612.138793}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.348765112948777, "sum": 1.348765112948777, "min": 1.348765112948777}}, "EndTime": 1552275612.138823, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552275612.138817}
    [0m
    [31m[03/11/2019 03:40:12 INFO 140575172896576] #quality_metric: host=algo-1, epoch=3, train binary_classification_weighted_cross_entropy_objective <loss>=0.376421118271[0m
    [31m[03/11/2019 03:40:12 INFO 140575172896576] #early_stopping_criteria_metric: host=algo-1, epoch=3, criteria=binary_classification_weighted_cross_entropy_objective, value=0.361575149421[0m
    [31m[03/11/2019 03:40:12 INFO 140575172896576] Epoch 3: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:40:12 INFO 140575172896576] #progress_metric: host=algo-1, completed 26 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 812, "sum": 812.0, "min": 812}, "Total Records Seen": {"count": 1, "max": 809456, "sum": 809456.0, "min": 809456}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 6, "sum": 6.0, "min": 6}}, "EndTime": 1552275612.141197, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1552275605.83637}
    [0m
    [31m[03/11/2019 03:40:12 INFO 140575172896576] #throughput_metric: host=algo-1, train throughput=31620.2600103 records/second[0m
    [31m[2019-03-11 03:40:12.141] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 5, "duration": 6304, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3651828865549672, "sum": 0.3651828865549672, "min": 0.3651828865549672}}, "EndTime": 1552275618.517088, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552275618.517023}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3535009060193546, "sum": 0.3535009060193546, "min": 0.3535009060193546}}, "EndTime": 1552275618.517163, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552275618.51715}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.36066419469172034, "sum": 0.36066419469172034, "min": 0.36066419469172034}}, "EndTime": 1552275618.517198, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552275618.51719}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3583622215309335, "sum": 0.3583622215309335, "min": 0.3583622215309335}}, "EndTime": 1552275618.517237, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552275618.517228}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.48255323714826576, "sum": 0.48255323714826576, "min": 0.48255323714826576}}, "EndTime": 1552275618.517272, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552275618.517264}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6404641866636036, "sum": 0.6404641866636036, "min": 0.6404641866636036}}, "EndTime": 1552275618.517305, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552275618.517297}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.48250414187944113, "sum": 0.48250414187944113, "min": 0.48250414187944113}}, "EndTime": 1552275618.517348, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552275618.51734}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6370406882127925, "sum": 0.6370406882127925, "min": 0.6370406882127925}}, "EndTime": 1552275618.517403, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552275618.517387}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.36155209388924603, "sum": 0.36155209388924603, "min": 0.36155209388924603}}, "EndTime": 1552275618.517457, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552275618.517442}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.36159848973259856, "sum": 0.36159848973259856, "min": 0.36159848973259856}}, "EndTime": 1552275618.517509, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552275618.517494}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3665388703466061, "sum": 0.3665388703466061, "min": 0.3665388703466061}}, "EndTime": 1552275618.517558, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552275618.517545}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3610365468700927, "sum": 0.3610365468700927, "min": 0.3610365468700927}}, "EndTime": 1552275618.5176, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552275618.51759}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.43677961008273175, "sum": 0.43677961008273175, "min": 0.43677961008273175}}, "EndTime": 1552275618.517652, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552275618.517637}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6353890059964741, "sum": 0.6353890059964741, "min": 0.6353890059964741}}, "EndTime": 1552275618.517703, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552275618.51769}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.48457892844425376, "sum": 0.48457892844425376, "min": 0.48457892844425376}}, "EndTime": 1552275618.517749, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552275618.517736}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6465855449695683, "sum": 0.6465855449695683, "min": 0.6465855449695683}}, "EndTime": 1552275618.517803, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552275618.517788}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5052605467369808, "sum": 0.5052605467369808, "min": 0.5052605467369808}}, "EndTime": 1552275618.517857, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552275618.517842}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5100626200767019, "sum": 0.5100626200767019, "min": 0.5100626200767019}}, "EndTime": 1552275618.517911, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552275618.517896}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5052216142050585, "sum": 0.5052216142050585, "min": 0.5052216142050585}}, "EndTime": 1552275618.517963, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552275618.517948}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5101361088585015, "sum": 0.5101361088585015, "min": 0.5101361088585015}}, "EndTime": 1552275618.518015, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552275618.518}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5432823336040554, "sum": 0.5432823336040554, "min": 0.5432823336040554}}, "EndTime": 1552275618.518067, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552275618.518053}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5973346039278423, "sum": 0.5973346039278423, "min": 0.5973346039278423}}, "EndTime": 1552275618.518124, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552275618.518108}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5431448340104453, "sum": 0.5431448340104453, "min": 0.5431448340104453}}, "EndTime": 1552275618.518177, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552275618.518162}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5972607800852713, "sum": 0.5972607800852713, "min": 0.5972607800852713}}, "EndTime": 1552275618.518231, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552275618.518216}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1660935969520454, "sum": 1.1660935969520454, "min": 1.1660935969520454}}, "EndTime": 1552275618.518282, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552275618.518268}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1656119823359963, "sum": 1.1656119823359963, "min": 1.1656119823359963}}, "EndTime": 1552275618.518326, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552275618.518315}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1659982738399026, "sum": 1.1659982738399026, "min": 1.1659982738399026}}, "EndTime": 1552275618.518373, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552275618.518361}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.165628966058319, "sum": 1.165628966058319, "min": 1.165628966058319}}, "EndTime": 1552275618.518422, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552275618.518409}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.2179688801981097, "sum": 1.2179688801981097, "min": 1.2179688801981097}}, "EndTime": 1552275618.518471, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552275618.518458}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.3417915374142442, "sum": 1.3417915374142442, "min": 1.3417915374142442}}, "EndTime": 1552275618.518521, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552275618.518507}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.2180785441182966, "sum": 1.2180785441182966, "min": 1.2180785441182966}}, "EndTime": 1552275618.518578, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552275618.518562}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.3412464424785058, "sum": 1.3412464424785058, "min": 1.3412464424785058}}, "EndTime": 1552275618.518627, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552275618.518616}
    [0m
    [31m[03/11/2019 03:40:18 INFO 140575172896576] #quality_metric: host=algo-1, epoch=4, train binary_classification_weighted_cross_entropy_objective <loss>=0.365182886555[0m
    [31m[03/11/2019 03:40:18 INFO 140575172896576] #early_stopping_criteria_metric: host=algo-1, epoch=4, criteria=binary_classification_weighted_cross_entropy_objective, value=0.353500906019[0m
    [31m[03/11/2019 03:40:18 INFO 140575172896576] Epoch 4: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:40:18 INFO 140575172896576] #progress_metric: host=algo-1, completed 33 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 1012, "sum": 1012.0, "min": 1012}, "Total Records Seen": {"count": 1, "max": 1008820, "sum": 1008820.0, "min": 1008820}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 7, "sum": 7.0, "min": 7}}, "EndTime": 1552275618.521897, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1552275612.141475}
    [0m
    [31m[03/11/2019 03:40:18 INFO 140575172896576] #throughput_metric: host=algo-1, train throughput=31245.6656248 records/second[0m
    [31m[2019-03-11 03:40:18.522] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 6, "duration": 6380, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.35806176627461034, "sum": 0.35806176627461034, "min": 0.35806176627461034}}, "EndTime": 1552275624.827147, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552275624.827087}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34855323484794576, "sum": 0.34855323484794576, "min": 0.34855323484794576}}, "EndTime": 1552275624.827222, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552275624.82721}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3543773801794004, "sum": 0.3543773801794004, "min": 0.3543773801794004}}, "EndTime": 1552275624.827273, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552275624.82726}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3522900390625, "sum": 0.3522900390625, "min": 0.3522900390625}}, "EndTime": 1552275624.82732, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552275624.827306}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.41642863010042275, "sum": 0.41642863010042275, "min": 0.41642863010042275}}, "EndTime": 1552275624.82736, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552275624.827351}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6387309703731058, "sum": 0.6387309703731058, "min": 0.6387309703731058}}, "EndTime": 1552275624.827388, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552275624.827381}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.4263452541792213, "sum": 0.4263452541792213, "min": 0.4263452541792213}}, "EndTime": 1552275624.827435, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552275624.827422}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6099737069786494, "sum": 0.6099737069786494, "min": 0.6099737069786494}}, "EndTime": 1552275624.827484, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552275624.827469}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.35555598238844366, "sum": 0.35555598238844366, "min": 0.35555598238844366}}, "EndTime": 1552275624.827548, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552275624.827531}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.355062516869013, "sum": 0.355062516869013, "min": 0.355062516869013}}, "EndTime": 1552275624.827606, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552275624.82759}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.35968572714340746, "sum": 0.35968572714340746, "min": 0.35968572714340746}}, "EndTime": 1552275624.827669, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552275624.827653}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.35472683363104585, "sum": 0.35472683363104585, "min": 0.35472683363104585}}, "EndTime": 1552275624.827733, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552275624.827717}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.4209054723384991, "sum": 0.4209054723384991, "min": 0.4209054723384991}}, "EndTime": 1552275624.827813, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552275624.827796}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6355837141257434, "sum": 0.6355837141257434, "min": 0.6355837141257434}}, "EndTime": 1552275624.827879, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552275624.827865}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.4188486150425283, "sum": 0.4188486150425283, "min": 0.4188486150425283}}, "EndTime": 1552275624.827933, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552275624.827918}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6536027886280462, "sum": 0.6536027886280462, "min": 0.6536027886280462}}, "EndTime": 1552275624.827973, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552275624.827964}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.503968337475954, "sum": 0.503968337475954, "min": 0.503968337475954}}, "EndTime": 1552275624.828, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552275624.827993}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5099149558675948, "sum": 0.5099149558675948, "min": 0.5099149558675948}}, "EndTime": 1552275624.82803, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552275624.828019}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5040184928855704, "sum": 0.5040184928855704, "min": 0.5040184928855704}}, "EndTime": 1552275624.828078, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552275624.828066}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5099665906896543, "sum": 0.5099665906896543, "min": 0.5099665906896543}}, "EndTime": 1552275624.828131, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552275624.828115}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5343341753015566, "sum": 0.5343341753015566, "min": 0.5343341753015566}}, "EndTime": 1552275624.828183, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552275624.828168}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5959916668034079, "sum": 0.5959916668034079, "min": 0.5959916668034079}}, "EndTime": 1552275624.828235, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552275624.82822}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5342394232055051, "sum": 0.5342394232055051, "min": 0.5342394232055051}}, "EndTime": 1552275624.828297, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552275624.828281}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5959276057487756, "sum": 0.5959276057487756, "min": 0.5959276057487756}}, "EndTime": 1552275624.828361, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552275624.828344}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.165481177861966, "sum": 1.165481177861966, "min": 1.165481177861966}}, "EndTime": 1552275624.828425, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552275624.828409}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.165601423253965, "sum": 1.165601423253965, "min": 1.165601423253965}}, "EndTime": 1552275624.828488, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552275624.828472}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1653950100232608, "sum": 1.1653950100232608, "min": 1.1653950100232608}}, "EndTime": 1552275624.828548, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552275624.828532}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1656148482279562, "sum": 1.1656148482279562, "min": 1.1656148482279562}}, "EndTime": 1552275624.828609, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552275624.828593}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1937079349690347, "sum": 1.1937079349690347, "min": 1.1937079349690347}}, "EndTime": 1552275624.828663, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552275624.828649}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.3341801171230911, "sum": 1.3341801171230911, "min": 1.3341801171230911}}, "EndTime": 1552275624.828715, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552275624.828702}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1937398098892902, "sum": 1.1937398098892902, "min": 1.1937398098892902}}, "EndTime": 1552275624.828769, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552275624.828753}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.3338634862851857, "sum": 1.3338634862851857, "min": 1.3338634862851857}}, "EndTime": 1552275624.828831, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552275624.828815}
    [0m
    [31m[03/11/2019 03:40:24 INFO 140575172896576] #quality_metric: host=algo-1, epoch=5, train binary_classification_weighted_cross_entropy_objective <loss>=0.358061766275[0m
    [31m[03/11/2019 03:40:24 INFO 140575172896576] #early_stopping_criteria_metric: host=algo-1, epoch=5, criteria=binary_classification_weighted_cross_entropy_objective, value=0.348553234848[0m
    [31m[03/11/2019 03:40:24 INFO 140575172896576] Epoch 5: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:40:24 INFO 140575172896576] #progress_metric: host=algo-1, completed 40 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 1212, "sum": 1212.0, "min": 1212}, "Total Records Seen": {"count": 1, "max": 1208184, "sum": 1208184.0, "min": 1208184}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 8, "sum": 8.0, "min": 8}}, "EndTime": 1552275624.83149, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1552275618.522097}
    [0m
    [31m[03/11/2019 03:40:24 INFO 140575172896576] #throughput_metric: host=algo-1, train throughput=31597.3824011 records/second[0m
    [31m[2019-03-11 03:40:24.831] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 7, "duration": 6309, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.35317101517873795, "sum": 0.35317101517873795, "min": 0.35317101517873795}}, "EndTime": 1552275631.06819, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552275631.06813}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34528309504830057, "sum": 0.34528309504830057, "min": 0.34528309504830057}}, "EndTime": 1552275631.068272, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552275631.068255}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3500301142936975, "sum": 0.3500301142936975, "min": 0.3500301142936975}}, "EndTime": 1552275631.06833, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552275631.068315}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34819938180434645, "sum": 0.34819938180434645, "min": 0.34819938180434645}}, "EndTime": 1552275631.068386, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552275631.068371}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3891295891766572, "sum": 0.3891295891766572, "min": 0.3891295891766572}}, "EndTime": 1552275631.068444, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552275631.068429}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6296487351614027, "sum": 0.6296487351614027, "min": 0.6296487351614027}}, "EndTime": 1552275631.068495, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552275631.068482}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.394590725731011, "sum": 0.394590725731011, "min": 0.394590725731011}}, "EndTime": 1552275631.068549, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552275631.068534}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6279934317814045, "sum": 0.6279934317814045, "min": 0.6279934317814045}}, "EndTime": 1552275631.068604, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552275631.068588}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.35141688480089656, "sum": 0.35141688480089656, "min": 0.35141688480089656}}, "EndTime": 1552275631.068659, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552275631.068643}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.35082043813580843, "sum": 0.35082043813580843, "min": 0.35082043813580843}}, "EndTime": 1552275631.068715, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552275631.0687}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.35497230564289955, "sum": 0.35497230564289955, "min": 0.35497230564289955}}, "EndTime": 1552275631.068771, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552275631.068756}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3506037244940523, "sum": 0.3506037244940523, "min": 0.3506037244940523}}, "EndTime": 1552275631.068837, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552275631.06882}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.39505894408872977, "sum": 0.39505894408872977, "min": 0.39505894408872977}}, "EndTime": 1552275631.068894, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552275631.068878}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6260122773443634, "sum": 0.6260122773443634, "min": 0.6260122773443634}}, "EndTime": 1552275631.06897, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552275631.068954}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.4093167758155708, "sum": 0.4093167758155708, "min": 0.4093167758155708}}, "EndTime": 1552275631.069027, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552275631.069012}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6923932212177832, "sum": 0.6923932212177832, "min": 0.6923932212177832}}, "EndTime": 1552275631.069089, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552275631.069072}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5028794399242306, "sum": 0.5028794399242306, "min": 0.5028794399242306}}, "EndTime": 1552275631.069147, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552275631.069131}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5097846717259392, "sum": 0.5097846717259392, "min": 0.5097846717259392}}, "EndTime": 1552275631.069201, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552275631.069186}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5029439985572394, "sum": 0.5029439985572394, "min": 0.5029439985572394}}, "EndTime": 1552275631.069253, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552275631.069239}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5098219938805355, "sum": 0.5098219938805355, "min": 0.5098219938805355}}, "EndTime": 1552275631.069308, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552275631.069292}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5277016542520955, "sum": 0.5277016542520955, "min": 0.5277016542520955}}, "EndTime": 1552275631.069363, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552275631.069347}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.594609693920193, "sum": 0.594609693920193, "min": 0.594609693920193}}, "EndTime": 1552275631.069425, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552275631.069409}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5276506366825583, "sum": 0.5276506366825583, "min": 0.5276506366825583}}, "EndTime": 1552275631.069479, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552275631.069464}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5945276401845654, "sum": 0.5945276401845654, "min": 0.5945276401845654}}, "EndTime": 1552275631.069533, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552275631.069518}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1646659637911236, "sum": 1.1646659637911236, "min": 1.1646659637911236}}, "EndTime": 1552275631.069576, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552275631.069563}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1655957068055118, "sum": 1.1655957068055118, "min": 1.1655957068055118}}, "EndTime": 1552275631.069628, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552275631.069614}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.164603540775165, "sum": 1.164603540775165, "min": 1.164603540775165}}, "EndTime": 1552275631.069682, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552275631.069667}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1656060886095516, "sum": 1.1656060886095516, "min": 1.1656060886095516}}, "EndTime": 1552275631.069744, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552275631.069729}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1840833206560144, "sum": 1.1840833206560144, "min": 1.1840833206560144}}, "EndTime": 1552275631.069804, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552275631.069789}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.3269847950384246, "sum": 1.3269847950384246, "min": 1.3269847950384246}}, "EndTime": 1552275631.069858, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552275631.069844}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1839885206366305, "sum": 1.1839885206366305, "min": 1.1839885206366305}}, "EndTime": 1552275631.069909, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552275631.069897}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.3267813469201477, "sum": 1.3267813469201477, "min": 1.3267813469201477}}, "EndTime": 1552275631.069939, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552275631.069932}
    [0m
    [31m[03/11/2019 03:40:31 INFO 140575172896576] #quality_metric: host=algo-1, epoch=6, train binary_classification_weighted_cross_entropy_objective <loss>=0.353171015179[0m
    [31m[03/11/2019 03:40:31 INFO 140575172896576] #early_stopping_criteria_metric: host=algo-1, epoch=6, criteria=binary_classification_weighted_cross_entropy_objective, value=0.345283095048[0m
    [31m[03/11/2019 03:40:31 INFO 140575172896576] Epoch 6: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:40:31 INFO 140575172896576] #progress_metric: host=algo-1, completed 46 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 1412, "sum": 1412.0, "min": 1412}, "Total Records Seen": {"count": 1, "max": 1407548, "sum": 1407548.0, "min": 1407548}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 9, "sum": 9.0, "min": 9}}, "EndTime": 1552275631.07254, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1552275624.831787}
    [0m
    [31m[03/11/2019 03:40:31 INFO 140575172896576] #throughput_metric: host=algo-1, train throughput=31944.7659997 records/second[0m
    [31m[2019-03-11 03:40:31.072] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 8, "duration": 6240, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.349640905178971, "sum": 0.349640905178971, "min": 0.349640905178971}}, "EndTime": 1552275637.791579, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552275637.791515}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34294272801384856, "sum": 0.34294272801384856, "min": 0.34294272801384856}}, "EndTime": 1552275637.79168, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552275637.791661}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34686880205624065, "sum": 0.34686880205624065, "min": 0.34686880205624065}}, "EndTime": 1552275637.791741, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552275637.791726}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34525132471592584, "sum": 0.34525132471592584, "min": 0.34525132471592584}}, "EndTime": 1552275637.791821, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552275637.791804}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.39339666817056473, "sum": 0.39339666817056473, "min": 0.39339666817056473}}, "EndTime": 1552275637.791873, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552275637.791859}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6260022582816119, "sum": 0.6260022582816119, "min": 0.6260022582816119}}, "EndTime": 1552275637.791929, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552275637.791913}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3888993842733565, "sum": 0.3888993842733565, "min": 0.3888993842733565}}, "EndTime": 1552275637.791994, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552275637.791977}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6040140958239685, "sum": 0.6040140958239685, "min": 0.6040140958239685}}, "EndTime": 1552275637.792052, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552275637.792036}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3484091053871653, "sum": 0.3484091053871653, "min": 0.3484091053871653}}, "EndTime": 1552275637.792113, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552275637.792097}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.347946627784614, "sum": 0.347946627784614, "min": 0.347946627784614}}, "EndTime": 1552275637.792171, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552275637.792154}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.35155894040821783, "sum": 0.35155894040821783, "min": 0.35155894040821783}}, "EndTime": 1552275637.79223, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552275637.792214}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3477796048495039, "sum": 0.3477796048495039, "min": 0.3477796048495039}}, "EndTime": 1552275637.792297, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552275637.792282}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3969215943226263, "sum": 0.3969215943226263, "min": 0.3969215943226263}}, "EndTime": 1552275637.792353, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552275637.792337}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6151315656307355, "sum": 0.6151315656307355, "min": 0.6151315656307355}}, "EndTime": 1552275637.792407, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552275637.792392}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3832322796002105, "sum": 0.3832322796002105, "min": 0.3832322796002105}}, "EndTime": 1552275637.792461, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552275637.792446}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6553429662618205, "sum": 0.6553429662618205, "min": 0.6553429662618205}}, "EndTime": 1552275637.792517, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552275637.792502}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5019372299424368, "sum": 0.5019372299424368, "min": 0.5019372299424368}}, "EndTime": 1552275637.792571, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552275637.792555}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5096615561480499, "sum": 0.5096615561480499, "min": 0.5096615561480499}}, "EndTime": 1552275637.79263, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552275637.792614}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5019936022734522, "sum": 0.5019936022734522, "min": 0.5019936022734522}}, "EndTime": 1552275637.792688, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552275637.792672}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5096888131759874, "sum": 0.5096888131759874, "min": 0.5096888131759874}}, "EndTime": 1552275637.792742, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552275637.792726}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5229926883563324, "sum": 0.5229926883563324, "min": 0.5229926883563324}}, "EndTime": 1552275637.792793, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552275637.792778}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5930843838447303, "sum": 0.5930843838447303, "min": 0.5930843838447303}}, "EndTime": 1552275637.792838, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552275637.792828}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5229712625628141, "sum": 0.5229712625628141, "min": 0.5229712625628141}}, "EndTime": 1552275637.792886, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552275637.792871}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5930094521872362, "sum": 0.5930094521872362, "min": 0.5930094521872362}}, "EndTime": 1552275637.792939, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552275637.792924}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1639128795221223, "sum": 1.1639128795221223, "min": 1.1639128795221223}}, "EndTime": 1552275637.792978, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552275637.792969}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1655941846071176, "sum": 1.1655941846071176, "min": 1.1655941846071176}}, "EndTime": 1552275637.793011, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552275637.792998}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1638703628616716, "sum": 1.1638703628616716, "min": 1.1638703628616716}}, "EndTime": 1552275637.793063, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552275637.793049}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.16560186368856, "sum": 1.16560186368856, "min": 1.16560186368856}}, "EndTime": 1552275637.79312, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552275637.793105}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1810852133592769, "sum": 1.1810852133592769, "min": 1.1810852133592769}}, "EndTime": 1552275637.793165, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552275637.793151}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.32025075182124, "sum": 1.32025075182124, "min": 1.32025075182124}}, "EndTime": 1552275637.79322, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552275637.793205}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1810465575558455, "sum": 1.1810465575558455, "min": 1.1810465575558455}}, "EndTime": 1552275637.793275, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552275637.793259}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.3201223543253375, "sum": 1.3201223543253375, "min": 1.3201223543253375}}, "EndTime": 1552275637.793339, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552275637.793322}
    [0m
    [31m[03/11/2019 03:40:37 INFO 140575172896576] #quality_metric: host=algo-1, epoch=7, train binary_classification_weighted_cross_entropy_objective <loss>=0.349640905179[0m
    [31m[03/11/2019 03:40:37 INFO 140575172896576] #early_stopping_criteria_metric: host=algo-1, epoch=7, criteria=binary_classification_weighted_cross_entropy_objective, value=0.342942728014[0m
    [31m[03/11/2019 03:40:37 INFO 140575172896576] Epoch 7: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:40:37 INFO 140575172896576] #progress_metric: host=algo-1, completed 53 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 1612, "sum": 1612.0, "min": 1612}, "Total Records Seen": {"count": 1, "max": 1606912, "sum": 1606912.0, "min": 1606912}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 10, "sum": 10.0, "min": 10}}, "EndTime": 1552275637.795983, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1552275631.072825}
    [0m
    [31m[03/11/2019 03:40:37 INFO 140575172896576] #throughput_metric: host=algo-1, train throughput=29652.7761087 records/second[0m
    [31m[2019-03-11 03:40:37.796] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 9, "duration": 6723, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.347006891145179, "sum": 0.347006891145179, "min": 0.347006891145179}}, "EndTime": 1552275644.033588, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552275644.033526}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3411967257782442, "sum": 0.3411967257782442, "min": 0.3411967257782442}}, "EndTime": 1552275644.033663, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552275644.033651}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3444969539163101, "sum": 0.3444969539163101, "min": 0.3444969539163101}}, "EndTime": 1552275644.033715, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552275644.0337}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34306995453187566, "sum": 0.34306995453187566, "min": 0.34306995453187566}}, "EndTime": 1552275644.033763, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552275644.033749}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.37010197816781665, "sum": 0.37010197816781665, "min": 0.37010197816781665}}, "EndTime": 1552275644.033801, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552275644.033792}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6173712888746405, "sum": 0.6173712888746405, "min": 0.6173712888746405}}, "EndTime": 1552275644.033834, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552275644.033822}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3686550999550364, "sum": 0.3686550999550364, "min": 0.3686550999550364}}, "EndTime": 1552275644.033884, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552275644.03387}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.619885236979729, "sum": 0.619885236979729, "min": 0.619885236979729}}, "EndTime": 1552275644.033934, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552275644.03392}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3461459124004422, "sum": 0.3461459124004422, "min": 0.3461459124004422}}, "EndTime": 1552275644.033994, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552275644.033979}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3458885993765826, "sum": 0.3458885993765826, "min": 0.3458885993765826}}, "EndTime": 1552275644.03406, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552275644.034043}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3490008349394679, "sum": 0.3490008349394679, "min": 0.3490008349394679}}, "EndTime": 1552275644.034125, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552275644.034109}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.345774281947457, "sum": 0.345774281947457, "min": 0.345774281947457}}, "EndTime": 1552275644.034189, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552275644.034173}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3744585236209122, "sum": 0.3744585236209122, "min": 0.3744585236209122}}, "EndTime": 1552275644.034261, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552275644.034246}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6302096776147583, "sum": 0.6302096776147583, "min": 0.6302096776147583}}, "EndTime": 1552275644.034322, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552275644.034306}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3754779644299991, "sum": 0.3754779644299991, "min": 0.3754779644299991}}, "EndTime": 1552275644.034382, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552275644.034367}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5735211174931358, "sum": 0.5735211174931358, "min": 0.5735211174931358}}, "EndTime": 1552275644.034443, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552275644.034427}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5011246051117404, "sum": 0.5011246051117404, "min": 0.5011246051117404}}, "EndTime": 1552275644.034502, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552275644.034487}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5095432196382302, "sum": 0.5095432196382302, "min": 0.5095432196382302}}, "EndTime": 1552275644.034554, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552275644.034541}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5011678960598893, "sum": 0.5011678960598893, "min": 0.5011678960598893}}, "EndTime": 1552275644.034601, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552275644.034588}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5095630672589019, "sum": 0.5095630672589019, "min": 0.5095630672589019}}, "EndTime": 1552275644.034653, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552275644.034639}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.51962202921345, "sum": 0.51962202921345, "min": 0.51962202921345}}, "EndTime": 1552275644.03471, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552275644.034695}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5914428989851295, "sum": 0.5914428989851295, "min": 0.5914428989851295}}, "EndTime": 1552275644.034772, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552275644.034755}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5196154634101906, "sum": 0.5196154634101906, "min": 0.5196154634101906}}, "EndTime": 1552275644.034826, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552275644.034811}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5913778166651127, "sum": 0.5913778166651127, "min": 0.5913778166651127}}, "EndTime": 1552275644.034888, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552275644.034872}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.163333273422778, "sum": 1.163333273422778, "min": 1.163333273422778}}, "EndTime": 1552275644.03494, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552275644.034928}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.165600453132361, "sum": 1.165600453132361, "min": 1.165600453132361}}, "EndTime": 1552275644.034994, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552275644.034978}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.16330359224099, "sum": 1.16330359224099, "min": 1.16330359224099}}, "EndTime": 1552275644.035047, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552275644.035032}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1656058300535883, "sum": 1.1656058300535883, "min": 1.1656058300535883}}, "EndTime": 1552275644.035099, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552275644.035084}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1798437187156485, "sum": 1.1798437187156485, "min": 1.1798437187156485}}, "EndTime": 1552275644.035158, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552275644.035142}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.313886191900052, "sum": 1.313886191900052, "min": 1.313886191900052}}, "EndTime": 1552275644.035209, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552275644.035197}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.179719946013024, "sum": 1.179719946013024, "min": 1.179719946013024}}, "EndTime": 1552275644.035262, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552275644.035246}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.313814590070715, "sum": 1.313814590070715, "min": 1.313814590070715}}, "EndTime": 1552275644.035312, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552275644.0353}
    [0m
    [31m[03/11/2019 03:40:44 INFO 140575172896576] #quality_metric: host=algo-1, epoch=8, train binary_classification_weighted_cross_entropy_objective <loss>=0.347006891145[0m
    [31m[03/11/2019 03:40:44 INFO 140575172896576] #early_stopping_criteria_metric: host=algo-1, epoch=8, criteria=binary_classification_weighted_cross_entropy_objective, value=0.341196725778[0m
    [31m[03/11/2019 03:40:44 INFO 140575172896576] Epoch 8: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:40:44 INFO 140575172896576] #progress_metric: host=algo-1, completed 60 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 1812, "sum": 1812.0, "min": 1812}, "Total Records Seen": {"count": 1, "max": 1806276, "sum": 1806276.0, "min": 1806276}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 11, "sum": 11.0, "min": 11}}, "EndTime": 1552275644.037981, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1552275637.796268}
    [0m
    [31m[03/11/2019 03:40:44 INFO 140575172896576] #throughput_metric: host=algo-1, train throughput=31939.9047714 records/second[0m
    [31m[2019-03-11 03:40:44.038] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 10, "duration": 6241, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34497469671048114, "sum": 0.34497469671048114, "min": 0.34497469671048114}}, "EndTime": 1552275649.948189, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552275649.948129}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.33982736052700024, "sum": 0.33982736052700024, "min": 0.33982736052700024}}, "EndTime": 1552275649.948262, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552275649.94825}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3426661689413253, "sum": 0.3426661689413253, "min": 0.3426661689413253}}, "EndTime": 1552275649.948313, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552275649.948298}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3414100283138716, "sum": 0.3414100283138716, "min": 0.3414100283138716}}, "EndTime": 1552275649.948362, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552275649.948351}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3568473606588852, "sum": 0.3568473606588852, "min": 0.3568473606588852}}, "EndTime": 1552275649.948392, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552275649.948385}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6148977909183981, "sum": 0.6148977909183981, "min": 0.6148977909183981}}, "EndTime": 1552275649.948428, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552275649.948416}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3576055860663179, "sum": 0.3576055860663179, "min": 0.3576055860663179}}, "EndTime": 1552275649.948478, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552275649.948464}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5979439923463754, "sum": 0.5979439923463754, "min": 0.5979439923463754}}, "EndTime": 1552275649.948527, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552275649.948513}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3444105252596601, "sum": 0.3444105252596601, "min": 0.3444105252596601}}, "EndTime": 1552275649.948576, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552275649.948562}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3443532883917267, "sum": 0.3443532883917267, "min": 0.3443532883917267}}, "EndTime": 1552275649.948628, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552275649.948613}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3470415938487604, "sum": 0.3470415938487604, "min": 0.3470415938487604}}, "EndTime": 1552275649.94868, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552275649.948666}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3442744213085079, "sum": 0.3442744213085079, "min": 0.3442744213085079}}, "EndTime": 1552275649.948735, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552275649.94872}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.36758389485421494, "sum": 0.36758389485421494, "min": 0.36758389485421494}}, "EndTime": 1552275649.948789, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552275649.948774}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6228373950018955, "sum": 0.6228373950018955, "min": 0.6228373950018955}}, "EndTime": 1552275649.948843, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552275649.948828}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3621376801303883, "sum": 0.3621376801303883, "min": 0.3621376801303883}}, "EndTime": 1552275649.948906, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552275649.948891}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6296973451010546, "sum": 0.6296973451010546, "min": 0.6296973451010546}}, "EndTime": 1552275649.948959, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552275649.948944}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.500431356075421, "sum": 0.500431356075421, "min": 0.500431356075421}}, "EndTime": 1552275649.949012, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552275649.948997}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5094262746686312, "sum": 0.5094262746686312, "min": 0.5094262746686312}}, "EndTime": 1552275649.949067, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552275649.949051}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.500462592426856, "sum": 0.500462592426856, "min": 0.500462592426856}}, "EndTime": 1552275649.949123, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552275649.949107}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5094407391572119, "sum": 0.5094407391572119, "min": 0.5094407391572119}}, "EndTime": 1552275649.949176, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552275649.949161}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5170382586148516, "sum": 0.5170382586148516, "min": 0.5170382586148516}}, "EndTime": 1552275649.949229, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552275649.949214}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5896801722924314, "sum": 0.5896801722924314, "min": 0.5896801722924314}}, "EndTime": 1552275649.949283, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552275649.949268}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5170357649146612, "sum": 0.5170357649146612, "min": 0.5170357649146612}}, "EndTime": 1552275649.949337, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552275649.949322}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5896412415624264, "sum": 0.5896412415624264, "min": 0.5896412415624264}}, "EndTime": 1552275649.94939, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552275649.949375}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1628935074542635, "sum": 1.1628935074542635, "min": 1.1628935074542635}}, "EndTime": 1552275649.949454, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552275649.949438}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1656102248915476, "sum": 1.1656102248915476, "min": 1.1656102248915476}}, "EndTime": 1552275649.949517, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552275649.949501}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1628731707855684, "sum": 1.1628731707855684, "min": 1.1628731707855684}}, "EndTime": 1552275649.949571, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552275649.949556}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.165613682425801, "sum": 1.165613682425801, "min": 1.165613682425801}}, "EndTime": 1552275649.949612, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552275649.9496}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1754669865747194, "sum": 1.1754669865747194, "min": 1.1754669865747194}}, "EndTime": 1552275649.949663, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552275649.94965}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.3078792055216266, "sum": 1.3078792055216266, "min": 1.3078792055216266}}, "EndTime": 1552275649.949724, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552275649.949708}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1753525574650596, "sum": 1.1753525574650596, "min": 1.1753525574650596}}, "EndTime": 1552275649.949776, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552275649.949761}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.3078432888625255, "sum": 1.3078432888625255, "min": 1.3078432888625255}}, "EndTime": 1552275649.949828, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552275649.949813}
    [0m
    [31m[03/11/2019 03:40:49 INFO 140575172896576] #quality_metric: host=algo-1, epoch=9, train binary_classification_weighted_cross_entropy_objective <loss>=0.34497469671[0m
    [31m[03/11/2019 03:40:49 INFO 140575172896576] #early_stopping_criteria_metric: host=algo-1, epoch=9, criteria=binary_classification_weighted_cross_entropy_objective, value=0.339827360527[0m
    [31m[03/11/2019 03:40:49 INFO 140575172896576] Epoch 9: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:40:49 INFO 140575172896576] #progress_metric: host=algo-1, completed 66 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 2012, "sum": 2012.0, "min": 2012}, "Total Records Seen": {"count": 1, "max": 2005640, "sum": 2005640.0, "min": 2005640}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 12, "sum": 12.0, "min": 12}}, "EndTime": 1552275649.952532, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1552275644.03827}
    [0m
    [31m[03/11/2019 03:40:49 INFO 140575172896576] #throughput_metric: host=algo-1, train throughput=33708.3171192 records/second[0m
    [31m[2019-03-11 03:40:49.952] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 11, "duration": 5914, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.343385628781726, "sum": 0.343385628781726, "min": 0.343385628781726}}, "EndTime": 1552275656.818481, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552275656.81842}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.33867450073855604, "sum": 0.33867450073855604, "min": 0.33867450073855604}}, "EndTime": 1552275656.818558, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552275656.818545}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34123276504439926, "sum": 0.34123276504439926, "min": 0.34123276504439926}}, "EndTime": 1552275656.818611, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552275656.818597}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34010360775281434, "sum": 0.34010360775281434, "min": 0.34010360775281434}}, "EndTime": 1552275656.818662, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552275656.818647}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3513394770789985, "sum": 0.3513394770789985, "min": 0.3513394770789985}}, "EndTime": 1552275656.818703, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552275656.818694}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6082766550342042, "sum": 0.6082766550342042, "min": 0.6082766550342042}}, "EndTime": 1552275656.818732, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552275656.818725}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.35022275608388626, "sum": 0.35022275608388626, "min": 0.35022275608388626}}, "EndTime": 1552275656.818781, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552275656.818767}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6111685697085893, "sum": 0.6111685697085893, "min": 0.6111685697085893}}, "EndTime": 1552275656.818832, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552275656.818817}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34304809462964236, "sum": 0.34304809462964236, "min": 0.34304809462964236}}, "EndTime": 1552275656.818872, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552275656.818862}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3431856944021867, "sum": 0.3431856944021867, "min": 0.3431856944021867}}, "EndTime": 1552275656.818924, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552275656.818909}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3455205237997237, "sum": 0.3455205237997237, "min": 0.3455205237997237}}, "EndTime": 1552275656.818975, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552275656.81896}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.343107326948463, "sum": 0.343107326948463, "min": 0.343107326948463}}, "EndTime": 1552275656.819025, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552275656.819014}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.35804022151621145, "sum": 0.35804022151621145, "min": 0.35804022151621145}}, "EndTime": 1552275656.819054, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552275656.819048}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5973278189232601, "sum": 0.5973278189232601, "min": 0.5973278189232601}}, "EndTime": 1552275656.819081, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552275656.819075}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.35920516699402777, "sum": 0.35920516699402777, "min": 0.35920516699402777}}, "EndTime": 1552275656.819107, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552275656.8191}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.553969459744554, "sum": 0.553969459744554, "min": 0.553969459744554}}, "EndTime": 1552275656.819162, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552275656.819148}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.4998431644152157, "sum": 0.4998431644152157, "min": 0.4998431644152157}}, "EndTime": 1552275656.819225, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552275656.819209}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5093097843956109, "sum": 0.5093097843956109, "min": 0.5093097843956109}}, "EndTime": 1552275656.819281, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552275656.819267}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.499864671716738, "sum": 0.499864671716738, "min": 0.499864671716738}}, "EndTime": 1552275656.819335, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552275656.81932}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5093203556693379, "sum": 0.5093203556693379, "min": 0.5093203556693379}}, "EndTime": 1552275656.819389, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552275656.819375}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5148661216850856, "sum": 0.5148661216850856, "min": 0.5148661216850856}}, "EndTime": 1552275656.819441, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552275656.819427}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5879017156859738, "sum": 0.5879017156859738, "min": 0.5879017156859738}}, "EndTime": 1552275656.819495, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552275656.819481}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5148665229375637, "sum": 0.5148665229375637, "min": 0.5148665229375637}}, "EndTime": 1552275656.819559, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552275656.819543}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5878681464075444, "sum": 0.5878681464075444, "min": 0.5878681464075444}}, "EndTime": 1552275656.81961, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552275656.819596}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1625382359183614, "sum": 1.1625382359183614, "min": 1.1625382359183614}}, "EndTime": 1552275656.819674, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552275656.819656}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1656226803573533, "sum": 1.1656226803573533, "min": 1.1656226803573533}}, "EndTime": 1552275656.819729, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552275656.819714}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1625255948934123, "sum": 1.1625255948934123, "min": 1.1625255948934123}}, "EndTime": 1552275656.819804, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552275656.819787}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1656246040382576, "sum": 1.1656246040382576, "min": 1.1656246040382576}}, "EndTime": 1552275656.819854, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552275656.819844}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1713371868804472, "sum": 1.1713371868804472, "min": 1.1713371868804472}}, "EndTime": 1552275656.819908, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552275656.819892}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.3022406913718985, "sum": 1.3022406913718985, "min": 1.3022406913718985}}, "EndTime": 1552275656.819965, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552275656.81995}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.17130506168538, "sum": 1.17130506168538, "min": 1.17130506168538}}, "EndTime": 1552275656.820017, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552275656.820005}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.3022192376678314, "sum": 1.3022192376678314, "min": 1.3022192376678314}}, "EndTime": 1552275656.820071, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552275656.820056}
    [0m
    [31m[03/11/2019 03:40:56 INFO 140575172896576] #quality_metric: host=algo-1, epoch=10, train binary_classification_weighted_cross_entropy_objective <loss>=0.343385628782[0m
    [31m[03/11/2019 03:40:56 INFO 140575172896576] #early_stopping_criteria_metric: host=algo-1, epoch=10, criteria=binary_classification_weighted_cross_entropy_objective, value=0.338674500739[0m
    [31m[03/11/2019 03:40:56 INFO 140575172896576] Epoch 10: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:40:56 INFO 140575172896576] #progress_metric: host=algo-1, completed 73 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 2212, "sum": 2212.0, "min": 2212}, "Total Records Seen": {"count": 1, "max": 2205004, "sum": 2205004.0, "min": 2205004}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 13, "sum": 13.0, "min": 13}}, "EndTime": 1552275656.822763, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1552275649.952809}
    [0m
    [31m[03/11/2019 03:40:56 INFO 140575172896576] #throughput_metric: host=algo-1, train throughput=29019.1299339 records/second[0m
    [31m[2019-03-11 03:40:56.822] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 12, "duration": 6870, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34214417086653975, "sum": 0.34214417086653975, "min": 0.34214417086653975}}, "EndTime": 1552275663.12019, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552275663.12013}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3377189544601057, "sum": 0.3377189544601057, "min": 0.3377189544601057}}, "EndTime": 1552275663.120264, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552275663.120251}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.340102334410701, "sum": 0.340102334410701, "min": 0.340102334410701}}, "EndTime": 1552275663.120317, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552275663.120302}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3389972463636542, "sum": 0.3389972463636542, "min": 0.3389972463636542}}, "EndTime": 1552275663.12037, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552275663.120356}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34368074595389053, "sum": 0.34368074595389053, "min": 0.34368074595389053}}, "EndTime": 1552275663.120426, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552275663.120412}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6051267347096199, "sum": 0.6051267347096199, "min": 0.6051267347096199}}, "EndTime": 1552275663.120481, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552275663.120466}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34431217971878436, "sum": 0.34431217971878436, "min": 0.34431217971878436}}, "EndTime": 1552275663.120535, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552275663.12052}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5922150619545175, "sum": 0.5922150619545175, "min": 0.5922150619545175}}, "EndTime": 1552275663.12059, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552275663.120575}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3419869154733629, "sum": 0.3419869154733629, "min": 0.3419869154733629}}, "EndTime": 1552275663.120646, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552275663.12063}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3422508027733271, "sum": 0.3422508027733271, "min": 0.3422508027733271}}, "EndTime": 1552275663.120711, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552275663.120694}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3443068614557161, "sum": 0.3443068614557161, "min": 0.3443068614557161}}, "EndTime": 1552275663.120769, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552275663.120754}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3421717679967832, "sum": 0.3421717679967832, "min": 0.3421717679967832}}, "EndTime": 1552275663.120825, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552275663.12081}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.35510520057103145, "sum": 0.35510520057103145, "min": 0.35510520057103145}}, "EndTime": 1552275663.120882, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552275663.120866}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.588998402753667, "sum": 0.588998402753667, "min": 0.588998402753667}}, "EndTime": 1552275663.120938, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552275663.120923}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.35120845434294273, "sum": 0.35120845434294273, "min": 0.35120845434294273}}, "EndTime": 1552275663.121002, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552275663.120977}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.628038518435991, "sum": 0.628038518435991, "min": 0.628038518435991}}, "EndTime": 1552275663.121058, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552275663.121043}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.4993475785758627, "sum": 0.4993475785758627, "min": 0.4993475785758627}}, "EndTime": 1552275663.121113, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552275663.121098}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5091943125509137, "sum": 0.5091943125509137, "min": 0.5091943125509137}}, "EndTime": 1552275663.121166, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552275663.121151}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.499361797236917, "sum": 0.499361797236917, "min": 0.499361797236917}}, "EndTime": 1552275663.121218, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552275663.121203}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5092020252170275, "sum": 0.5092020252170275, "min": 0.5092020252170275}}, "EndTime": 1552275663.12127, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552275663.121256}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5131318281164121, "sum": 0.5131318281164121, "min": 0.5131318281164121}}, "EndTime": 1552275663.121325, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552275663.121309}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5861035508198954, "sum": 0.5861035508198954, "min": 0.5861035508198954}}, "EndTime": 1552275663.121377, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552275663.121362}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5131352119637494, "sum": 0.5131352119637494, "min": 0.5131352119637494}}, "EndTime": 1552275663.121433, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552275663.121418}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.586079047044917, "sum": 0.586079047044917, "min": 0.586079047044917}}, "EndTime": 1552275663.121492, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552275663.121476}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1622375295054372, "sum": 1.1622375295054372, "min": 1.1622375295054372}}, "EndTime": 1552275663.121542, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552275663.121528}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1656412175624216, "sum": 1.1656412175624216, "min": 1.1656412175624216}}, "EndTime": 1552275663.121595, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552275663.12158}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1622304077148438, "sum": 1.1622304077148438, "min": 1.1622304077148438}}, "EndTime": 1552275663.121647, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552275663.121633}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1656419435434007, "sum": 1.1656419435434007, "min": 1.1656419435434007}}, "EndTime": 1552275663.121694, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552275663.12168}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1690166780864772, "sum": 1.1690166780864772, "min": 1.1690166780864772}}, "EndTime": 1552275663.12174, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552275663.121726}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.2969777703213332, "sum": 1.2969777703213332, "min": 1.2969777703213332}}, "EndTime": 1552275663.121792, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552275663.121777}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1689940323566073, "sum": 1.1689940323566073, "min": 1.1689940323566073}}, "EndTime": 1552275663.121847, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552275663.121832}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.2969597775253219, "sum": 1.2969597775253219, "min": 1.2969597775253219}}, "EndTime": 1552275663.121902, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552275663.121887}
    [0m
    [31m[03/11/2019 03:41:03 INFO 140575172896576] #quality_metric: host=algo-1, epoch=11, train binary_classification_weighted_cross_entropy_objective <loss>=0.342144170867[0m
    [31m[03/11/2019 03:41:03 INFO 140575172896576] #early_stopping_criteria_metric: host=algo-1, epoch=11, criteria=binary_classification_weighted_cross_entropy_objective, value=0.33771895446[0m
    [31m[03/11/2019 03:41:03 INFO 140575172896576] Epoch 11: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:41:03 INFO 140575172896576] #progress_metric: host=algo-1, completed 80 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 2412, "sum": 2412.0, "min": 2412}, "Total Records Seen": {"count": 1, "max": 2404368, "sum": 2404368.0, "min": 2404368}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 14, "sum": 14.0, "min": 14}}, "EndTime": 1552275663.124508, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1552275656.823056}
    [0m
    [31m[03/11/2019 03:41:03 INFO 140575172896576] #throughput_metric: host=algo-1, train throughput=31637.209847 records/second[0m
    [31m[2019-03-11 03:41:03.124] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 13, "duration": 6301, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34114156809284457, "sum": 0.34114156809284457, "min": 0.34114156809284457}}, "EndTime": 1552275669.539862, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552275669.539799}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3368815071451005, "sum": 0.3368815071451005, "min": 0.3368815071451005}}, "EndTime": 1552275669.539938, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552275669.539926}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.33919448994392126, "sum": 0.33919448994392126, "min": 0.33919448994392126}}, "EndTime": 1552275669.539991, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552275669.539978}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3380386825638201, "sum": 0.3380386825638201, "min": 0.3380386825638201}}, "EndTime": 1552275669.54004, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552275669.540025}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.33985611253287923, "sum": 0.33985611253287923, "min": 0.33985611253287923}}, "EndTime": 1552275669.540076, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552275669.540067}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.598893346757745, "sum": 0.598893346757745, "min": 0.598893346757745}}, "EndTime": 1552275669.540106, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552275669.540098}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3390653486395601, "sum": 0.3390653486395601, "min": 0.3390653486395601}}, "EndTime": 1552275669.540156, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552275669.540142}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6011157977041887, "sum": 0.6011157977041887, "min": 0.6011157977041887}}, "EndTime": 1552275669.540204, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552275669.540191}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34112556994260856, "sum": 0.34112556994260856, "min": 0.34112556994260856}}, "EndTime": 1552275669.540253, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552275669.540239}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.341461145142215, "sum": 0.341461145142215, "min": 0.341461145142215}}, "EndTime": 1552275669.540292, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552275669.540283}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3433391585613615, "sum": 0.3433391585613615, "min": 0.3433391585613615}}, "EndTime": 1552275669.54032, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552275669.540313}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3414039370666197, "sum": 0.3414039370666197, "min": 0.3414039370666197}}, "EndTime": 1552275669.540352, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552275669.54034}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34829548959396595, "sum": 0.34829548959396595, "min": 0.34829548959396595}}, "EndTime": 1552275669.540402, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552275669.540388}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5974588073270405, "sum": 0.5974588073270405, "min": 0.5974588073270405}}, "EndTime": 1552275669.540448, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552275669.540438}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34793724305426055, "sum": 0.34793724305426055, "min": 0.34793724305426055}}, "EndTime": 1552275669.540476, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552275669.540469}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5641578885945843, "sum": 0.5641578885945843, "min": 0.5641578885945843}}, "EndTime": 1552275669.540502, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552275669.540495}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.4989315968422434, "sum": 0.4989315968422434, "min": 0.4989315968422434}}, "EndTime": 1552275669.540537, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552275669.540525}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5090775651787993, "sum": 0.5090775651787993, "min": 0.5090775651787993}}, "EndTime": 1552275669.5406, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552275669.540584}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.49894058357890525, "sum": 0.49894058357890525, "min": 0.49894058357890525}}, "EndTime": 1552275669.540656, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552275669.540641}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.509083063020179, "sum": 0.509083063020179, "min": 0.509083063020179}}, "EndTime": 1552275669.540711, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552275669.540696}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5118569437151578, "sum": 0.5118569437151578, "min": 0.5118569437151578}}, "EndTime": 1552275669.540767, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552275669.540752}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5843220768837474, "sum": 0.5843220768837474, "min": 0.5843220768837474}}, "EndTime": 1552275669.540822, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552275669.540807}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5118611776073974, "sum": 0.5118611776073974, "min": 0.5118611776073974}}, "EndTime": 1552275669.540876, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552275669.540862}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5843062766568745, "sum": 0.5843062766568745, "min": 0.5843062766568745}}, "EndTime": 1552275669.540929, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552275669.540915}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1619775651327928, "sum": 1.1619775651327928, "min": 1.1619775651327928}}, "EndTime": 1552275669.540991, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552275669.540975}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1656637395350777, "sum": 1.1656637395350777, "min": 1.1656637395350777}}, "EndTime": 1552275669.541041, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552275669.541031}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.161973793834897, "sum": 1.161973793834897, "min": 1.161973793834897}}, "EndTime": 1552275669.54107, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552275669.541063}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1656635217714548, "sum": 1.1656635217714548, "min": 1.1656635217714548}}, "EndTime": 1552275669.541125, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552275669.541109}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1673277857794833, "sum": 1.1673277857794833, "min": 1.1673277857794833}}, "EndTime": 1552275669.541181, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552275669.541166}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.292114694336551, "sum": 1.292114694336551, "min": 1.292114694336551}}, "EndTime": 1552275669.541237, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552275669.541222}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1673235492035372, "sum": 1.1673235492035372, "min": 1.1673235492035372}}, "EndTime": 1552275669.5413, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552275669.541284}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.2920980631766008, "sum": 1.2920980631766008, "min": 1.2920980631766008}}, "EndTime": 1552275669.541355, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552275669.54134}
    [0m
    [31m[03/11/2019 03:41:09 INFO 140575172896576] #quality_metric: host=algo-1, epoch=12, train binary_classification_weighted_cross_entropy_objective <loss>=0.341141568093[0m
    [31m[03/11/2019 03:41:09 INFO 140575172896576] #early_stopping_criteria_metric: host=algo-1, epoch=12, criteria=binary_classification_weighted_cross_entropy_objective, value=0.336881507145[0m
    [31m[03/11/2019 03:41:09 INFO 140575172896576] Epoch 12: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:41:09 INFO 140575172896576] #progress_metric: host=algo-1, completed 86 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 2612, "sum": 2612.0, "min": 2612}, "Total Records Seen": {"count": 1, "max": 2603732, "sum": 2603732.0, "min": 2603732}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 15, "sum": 15.0, "min": 15}}, "EndTime": 1552275669.544013, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1552275663.124773}
    [0m
    [31m[03/11/2019 03:41:09 INFO 140575172896576] #throughput_metric: host=algo-1, train throughput=31056.7031455 records/second[0m
    [31m[2019-03-11 03:41:09.544] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 14, "duration": 6419, "num_examples": 200}[0m
    
    2019-03-11 03:41:26 Uploading - Uploading generated training model[31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3403380548678451, "sum": 0.3403380548678451, "min": 0.3403380548678451}}, "EndTime": 1552275675.588339, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552275675.588277}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3361356635165574, "sum": 0.3361356635165574, "min": 0.3361356635165574}}, "EndTime": 1552275675.588418, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552275675.588404}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3384665114819704, "sum": 0.3384665114819704, "min": 0.3384665114819704}}, "EndTime": 1552275675.588473, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552275675.588457}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3372001446671222, "sum": 0.3372001446671222, "min": 0.3372001446671222}}, "EndTime": 1552275675.588533, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552275675.588518}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3345176142016847, "sum": 0.3345176142016847, "min": 0.3345176142016847}}, "EndTime": 1552275675.588589, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552275675.588574}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5952526120708216, "sum": 0.5952526120708216, "min": 0.5952526120708216}}, "EndTime": 1552275675.588645, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552275675.588629}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3348870686286658, "sum": 0.3348870686286658, "min": 0.3348870686286658}}, "EndTime": 1552275675.588701, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552275675.588685}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5863257333956771, "sum": 0.5863257333956771, "min": 0.5863257333956771}}, "EndTime": 1552275675.588757, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552275675.588741}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34043442167349197, "sum": 0.34043442167349197, "min": 0.34043442167349197}}, "EndTime": 1552275675.588811, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552275675.588796}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3407913978231612, "sum": 0.3407913978231612, "min": 0.3407913978231612}}, "EndTime": 1552275675.588866, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552275675.588851}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3425602123701393, "sum": 0.3425602123701393, "min": 0.3425602123701393}}, "EndTime": 1552275675.588906, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552275675.588892}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.340736872073993, "sum": 0.340736872073993, "min": 0.340736872073993}}, "EndTime": 1552275675.588949, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552275675.58894}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3451417777670089, "sum": 0.3451417777670089, "min": 0.3451417777670089}}, "EndTime": 1552275675.589004, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552275675.58899}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5988669708098598, "sum": 0.5988669708098598, "min": 0.5988669708098598}}, "EndTime": 1552275675.589056, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552275675.589041}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34307314653252835, "sum": 0.34307314653252835, "min": 0.34307314653252835}}, "EndTime": 1552275675.589108, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552275675.589094}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.6002385617356804, "sum": 0.6002385617356804, "min": 0.6002385617356804}}, "EndTime": 1552275675.589143, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552275675.589131}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.49858488594706935, "sum": 0.49858488594706935, "min": 0.49858488594706935}}, "EndTime": 1552275675.589195, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552275675.58918}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5089589788542321, "sum": 0.5089589788542321, "min": 0.5089589788542321}}, "EndTime": 1552275675.589251, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552275675.589236}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.49859018542419126, "sum": 0.49859018542419126, "min": 0.49859018542419126}}, "EndTime": 1552275675.589313, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552275675.589297}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5089629158518422, "sum": 0.5089629158518422, "min": 0.5089629158518422}}, "EndTime": 1552275675.589375, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552275675.589359}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5109003112639614, "sum": 0.5109003112639614, "min": 0.5109003112639614}}, "EndTime": 1552275675.589429, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552275675.589414}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5825426100151023, "sum": 0.5825426100151023, "min": 0.5825426100151023}}, "EndTime": 1552275675.58949, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552275675.589474}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5109039870219015, "sum": 0.5109039870219015, "min": 0.5109039870219015}}, "EndTime": 1552275675.58955, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552275675.589535}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5825327654507891, "sum": 0.5825327654507891, "min": 0.5825327654507891}}, "EndTime": 1552275675.589603, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552275675.589588}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1617488743767665, "sum": 1.1617488743767665, "min": 1.1617488743767665}}, "EndTime": 1552275675.589653, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552275675.589638}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1656883833228644, "sum": 1.1656883833228644, "min": 1.1656883833228644}}, "EndTime": 1552275675.589704, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552275675.589691}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1617468632837036, "sum": 1.1617468632837036, "min": 1.1617468632837036}}, "EndTime": 1552275675.589734, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552275675.589727}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1656874509265076, "sum": 1.1656874509265076, "min": 1.1656874509265076}}, "EndTime": 1552275675.589763, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552275675.589753}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1660039160646984, "sum": 1.1660039160646984, "min": 1.1660039160646984}}, "EndTime": 1552275675.589814, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552275675.589799}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.2875859535255625, "sum": 1.2875859535255625, "min": 1.2875859535255625}}, "EndTime": 1552275675.589868, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552275675.589853}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1660032005118366, "sum": 1.1660032005118366, "min": 1.1660032005118366}}, "EndTime": 1552275675.58993, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552275675.589914}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.2875715526791673, "sum": 1.2875715526791673, "min": 1.2875715526791673}}, "EndTime": 1552275675.589992, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552275675.589976}
    [0m
    [31m[03/11/2019 03:41:15 INFO 140575172896576] #quality_metric: host=algo-1, epoch=13, train binary_classification_weighted_cross_entropy_objective <loss>=0.340338054868[0m
    [31m[03/11/2019 03:41:15 INFO 140575172896576] #early_stopping_criteria_metric: host=algo-1, epoch=13, criteria=binary_classification_weighted_cross_entropy_objective, value=0.334517614202[0m
    [31m[03/11/2019 03:41:15 INFO 140575172896576] Epoch 13: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:41:15 INFO 140575172896576] #progress_metric: host=algo-1, completed 93 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 2812, "sum": 2812.0, "min": 2812}, "Total Records Seen": {"count": 1, "max": 2803096, "sum": 2803096.0, "min": 2803096}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 16, "sum": 16.0, "min": 16}}, "EndTime": 1552275675.592671, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1552275669.544276}
    [0m
    [31m[03/11/2019 03:41:15 INFO 140575172896576] #throughput_metric: host=algo-1, train throughput=32960.7691398 records/second[0m
    [31m[2019-03-11 03:41:15.592] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 15, "duration": 6048, "num_examples": 200}[0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.33968455390355096, "sum": 0.33968455390355096, "min": 0.33968455390355096}}, "EndTime": 1552275681.451495, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552275681.451436}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.33545819808729926, "sum": 0.33545819808729926, "min": 0.33545819808729926}}, "EndTime": 1552275681.451569, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552275681.451557}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.337879189074339, "sum": 0.337879189074339, "min": 0.337879189074339}}, "EndTime": 1552275681.451621, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552275681.451609}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3364432534452659, "sum": 0.3364432534452659, "min": 0.3364432534452659}}, "EndTime": 1552275681.45167, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552275681.45166}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3313742982107191, "sum": 0.3313742982107191, "min": 0.3313742982107191}}, "EndTime": 1552275681.451709, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552275681.451697}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5894885120296, "sum": 0.5894885120296, "min": 0.5894885120296}}, "EndTime": 1552275681.451757, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552275681.451743}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3312075480935562, "sum": 0.3312075480935562, "min": 0.3312075480935562}}, "EndTime": 1552275681.451845, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552275681.451827}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5907595437974786, "sum": 0.5907595437974786, "min": 0.5907595437974786}}, "EndTime": 1552275681.451912, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552275681.451895}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3398797196047989, "sum": 0.3398797196047989, "min": 0.3398797196047989}}, "EndTime": 1552275681.451976, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552275681.45196}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3401863389135006, "sum": 0.3401863389135006, "min": 0.3401863389135006}}, "EndTime": 1552275681.45204, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552275681.452024}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34193371179475257, "sum": 0.34193371179475257, "min": 0.34193371179475257}}, "EndTime": 1552275681.452103, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552275681.452086}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.3401562111245927, "sum": 0.3401562111245927, "min": 0.3401562111245927}}, "EndTime": 1552275681.452176, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552275681.45216}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34104891595888376, "sum": 0.34104891595888376, "min": 0.34104891595888376}}, "EndTime": 1552275681.452237, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552275681.452221}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5775458740541084, "sum": 0.5775458740541084, "min": 0.5775458740541084}}, "EndTime": 1552275681.45229, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552275681.452275}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.34031021612732854, "sum": 0.34031021612732854, "min": 0.34031021612732854}}, "EndTime": 1552275681.45234, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552275681.452326}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5386363755422621, "sum": 0.5386363755422621, "min": 0.5386363755422621}}, "EndTime": 1552275681.45242, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552275681.452403}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.4982975454665908, "sum": 0.4982975454665908, "min": 0.4982975454665908}}, "EndTime": 1552275681.452473, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552275681.452458}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.508840274293219, "sum": 0.508840274293219, "min": 0.508840274293219}}, "EndTime": 1552275681.452535, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552275681.452519}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.4983002771732196, "sum": 0.4983002771732196, "min": 0.4983002771732196}}, "EndTime": 1552275681.452592, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552275681.452576}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5088429926579922, "sum": 0.5088429926579922, "min": 0.5088429926579922}}, "EndTime": 1552275681.452657, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552275681.452637}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.510074233548725, "sum": 0.510074233548725, "min": 0.510074233548725}}, "EndTime": 1552275681.452723, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552275681.452705}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5808202184744217, "sum": 0.5808202184744217, "min": 0.5808202184744217}}, "EndTime": 1552275681.452786, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552275681.45277}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5100766696642391, "sum": 0.5100766696642391, "min": 0.5100766696642391}}, "EndTime": 1552275681.452841, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552275681.452826}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 0.5808142933486095, "sum": 0.5808142933486095, "min": 0.5808142933486095}}, "EndTime": 1552275681.452895, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552275681.45288}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1615434646989833, "sum": 1.1615434646989833, "min": 1.1615434646989833}}, "EndTime": 1552275681.452959, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552275681.452942}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1657150526190523, "sum": 1.1657150526190523, "min": 1.1657150526190523}}, "EndTime": 1552275681.453015, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552275681.453}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1615423191396435, "sum": 1.1615423191396435, "min": 1.1615423191396435}}, "EndTime": 1552275681.453076, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552275681.45306}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1657136181395258, "sum": 1.1657136181395258, "min": 1.1657136181395258}}, "EndTime": 1552275681.45314, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552275681.453123}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1653917190321725, "sum": 1.1653917190321725, "min": 1.1653917190321725}}, "EndTime": 1552275681.453194, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552275681.453179}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.2832254080460899, "sum": 1.2832254080460899, "min": 1.2832254080460899}}, "EndTime": 1552275681.453253, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552275681.453236}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.1653905458689935, "sum": 1.1653905458689935, "min": 1.1653905458689935}}, "EndTime": 1552275681.453315, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552275681.453299}
    [0m
    [31m#metrics {"Metrics": {"train_binary_classification_weighted_cross_entropy_objective": {"count": 1, "max": 1.283214558050261, "sum": 1.283214558050261, "min": 1.283214558050261}}, "EndTime": 1552275681.45337, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552275681.453355}
    [0m
    [31m[03/11/2019 03:41:21 INFO 140575172896576] #quality_metric: host=algo-1, epoch=14, train binary_classification_weighted_cross_entropy_objective <loss>=0.339684553904[0m
    [31m[03/11/2019 03:41:21 INFO 140575172896576] #early_stopping_criteria_metric: host=algo-1, epoch=14, criteria=binary_classification_weighted_cross_entropy_objective, value=0.331207548094[0m
    [31m[03/11/2019 03:41:21 INFO 140575172896576] Epoch 14: Loss improved. Updating best model[0m
    [31m[03/11/2019 03:41:21 INFO 140575172896576] #progress_metric: host=algo-1, completed 100 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Batches Since Last Reset": {"count": 1, "max": 200, "sum": 200.0, "min": 200}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 3012, "sum": 3012.0, "min": 3012}, "Total Records Seen": {"count": 1, "max": 3002460, "sum": 3002460.0, "min": 3002460}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 17, "sum": 17.0, "min": 17}}, "EndTime": 1552275681.456392, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1552275675.592932}
    [0m
    [31m[03/11/2019 03:41:21 INFO 140575172896576] #throughput_metric: host=algo-1, train throughput=34000.3197835 records/second[0m
    [31m[03/11/2019 03:41:21 WARNING 140575172896576] wait_for_all_workers will not sync workers since the kv store is not running distributed[0m
    [31m[03/11/2019 03:41:21 WARNING 140575172896576] wait_for_all_workers will not sync workers since the kv store is not running distributed[0m
    [31m[2019-03-11 03:41:21.456] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 16, "duration": 5863, "num_examples": 200}[0m
    [31m[2019-03-11 03:41:21.464] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 17, "duration": 7, "num_examples": 1}[0m
    [31m[2019-03-11 03:41:22.212] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 18, "duration": 743, "num_examples": 200}[0m
    [31m[03/11/2019 03:41:22 INFO 140575172896576] #train_score (algo-1) : ('binary_classification_weighted_cross_entropy_objective', 0.32092717971047102)[0m
    [31m[03/11/2019 03:41:22 INFO 140575172896576] #train_score (algo-1) : ('binary_classification_accuracy', 0.99933287855380104)[0m
    [31m[03/11/2019 03:41:22 INFO 140575172896576] #train_score (algo-1) : ('binary_f_1.000', 0.8041237113402062)[0m
    [31m[03/11/2019 03:41:22 INFO 140575172896576] #train_score (algo-1) : ('precision', 0.8323170731707317)[0m
    [31m[03/11/2019 03:41:22 INFO 140575172896576] #train_score (algo-1) : ('recall', 0.7777777777777778)[0m
    [31m[03/11/2019 03:41:22 INFO 140575172896576] #quality_metric: host=algo-1, train binary_classification_weighted_cross_entropy_objective <loss>=0.32092717971[0m
    [31m[03/11/2019 03:41:22 INFO 140575172896576] #quality_metric: host=algo-1, train binary_classification_accuracy <score>=0.999332878554[0m
    [31m[03/11/2019 03:41:22 INFO 140575172896576] #quality_metric: host=algo-1, train binary_f_1.000 <score>=0.80412371134[0m
    [31m[03/11/2019 03:41:22 INFO 140575172896576] #quality_metric: host=algo-1, train precision <score>=0.832317073171[0m
    [31m[03/11/2019 03:41:22 INFO 140575172896576] #quality_metric: host=algo-1, train recall <score>=0.777777777778[0m
    [31m[03/11/2019 03:41:22 INFO 140575172896576] Best model found for hyperparameters: {"lr_scheduler_step": 10, "wd": 0.0001, "optimizer": "adam", "lr_scheduler_factor": 0.99, "l1": 0.0, "learning_rate": 0.1, "lr_scheduler_minimum_lr": 0.0001}[0m
    [31m[03/11/2019 03:41:22 INFO 140575172896576] Saved checkpoint to "/tmp/tmpUx0FFH/mx-mod-0000.params"[0m
    [31m[03/11/2019 03:41:22 INFO 140575172896576] Test data is not provided.[0m
    [31m[2019-03-11 03:41:22.923] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 19, "duration": 710, "num_examples": 200}[0m
    [31m[2019-03-11 03:41:22.923] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "duration": 96476, "num_epochs": 20, "num_examples": 3413}[0m
    [31m#metrics {"Metrics": {"totaltime": {"count": 1, "max": 96727.1740436554, "sum": 96727.1740436554, "min": 96727.1740436554}, "finalize.time": {"count": 1, "max": 1454.1339874267578, "sum": 1454.1339874267578, "min": 1454.1339874267578}, "initialize.time": {"count": 1, "max": 237.81704902648926, "sum": 237.81704902648926, "min": 237.81704902648926}, "check_early_stopping.time": {"count": 15, "max": 1.0662078857421875, "sum": 12.913942337036133, "min": 0.7669925689697266}, "setuptime": {"count": 1, "max": 13.484001159667969, "sum": 13.484001159667969, "min": 13.484001159667969}, "update.time": {"count": 15, "max": 6869.810104370117, "sum": 94938.04788589478, "min": 5863.276958465576}, "epochs": {"count": 1, "max": 15, "sum": 15.0, "min": 15}}, "EndTime": 1552275682.923805, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1552275586.269268}
    [0m
    
    2019-03-11 03:41:33 Completed - Training job completed
    Billable seconds: 122
    CPU times: user 685 ms, sys: 17 ms, total: 702 ms
    Wall time: 5min 13s


This model trains for a fixed precision of 90%, and, under that constraint, tries to get as high a recall as possible.


```python
%%time 
# deploy and evaluate a predictor
precision_predictor = linear_precision.deploy(initial_instance_count=1, instance_type='ml.t2.medium')
```

    INFO:sagemaker:Creating model with name: linear-learner-2019-03-11-04-07-10-993
    INFO:sagemaker:Creating endpoint with name linear-learner-2019-03-11-03-36-56-524


    --------------------------------------------------------------------------!CPU times: user 380 ms, sys: 24.3 ms, total: 404 ms
    Wall time: 6min 15s



```python
print('Metrics for tuned (precision), LinearLearner.\n')

# get metrics for balanced predictor
metrics = evaluate(precision_predictor, 
                   test_features.astype('float32'), 
                   test_labels, 
                   verbose=True)
```

    Metrics for tuned (precision), LinearLearner.
    
    prediction (col)    0.0  1.0
    actual (row)                
    0.0               85276   26
    1.0                  31  110
    
    Recall:     0.780
    Precision:  0.809
    Accuracy:   0.999
    



```python
## IMPORTANT
# delete the predictor endpoint 
delete_endpoint(precision_predictor)
```

    Deleted linear-learner-2019-03-11-03-36-56-524


## Final Cleanup!

* Double check that you have deleted all your endpoints.
* I'd also suggest manually deleting your S3 bucket, models, and endpoint configurations directly from your AWS console.

You can find thorough cleanup instructions, [in the documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/ex1-cleanup.html).


