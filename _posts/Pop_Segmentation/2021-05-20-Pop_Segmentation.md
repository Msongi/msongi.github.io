# Population Segmentation with SageMaker

In this notebook, you'll employ two, unsupervised learning algorithms to do **population segmentation**. Population segmentation aims to find natural groupings in population data that reveal some feature-level similarities between different regions in the US.

Using **principal component analysis** (PCA) you will reduce the dimensionality of the original census data. Then, you'll use **k-means clustering** to assign each US county to a particular cluster based on where a county lies in component space. How each cluster is arranged in component space can tell you which US counties are most similar and what demographic traits define that similarity; this information is most often used to inform targeted, marketing campaigns that want to appeal to a specific group of people. This cluster information is also useful for learning more about a population by revealing patterns between regions that you otherwise may not have noticed.

### US Census Data

You'll be using data collected by the [US Census](https://en.wikipedia.org/wiki/United_States_Census), which aims to count the US population, recording demographic traits about labor, age, population, and so on, for each county in the US. The bulk of this notebook was taken from an existing SageMaker example notebook and [blog post](https://aws.amazon.com/blogs/machine-learning/analyze-us-census-data-for-population-segmentation-using-amazon-sagemaker/), and I've broken it down further into demonstrations and exercises for you to complete.

### Machine Learning Workflow

To implement population segmentation, you'll go through a number of steps:
* Data loading and exploration
* Data cleaning and pre-processing 
* Dimensionality reduction with PCA
* Feature engineering and data transformation
* Clustering transformed data with k-means
* Extracting trained model attributes and visualizing k clusters

These tasks make up a complete, machine learning workflow from data loading and cleaning to model deployment. Each exercise is designed to give you practice with part of the machine learning workflow, and to demonstrate how to use SageMaker tools, such as built-in data management with S3 and built-in algorithms.

---

First, import the relevant libraries into this SageMaker notebook. 


```python
# data managing and display libs
import pandas as pd
import numpy as np
import os
import io

import matplotlib.pyplot as plt
import matplotlib
%matplotlib inline 
```


```python
# sagemaker libraries
import boto3
import sagemaker
```

## Loading the Data from Amazon S3

This particular dataset is already in an Amazon S3 bucket; you can load the data by pointing to this bucket and getting a data file by name. 

> You can interact with S3 using a `boto3` client.


```python
# boto3 client to get S3 data
s3_client = boto3.client('s3')
bucket_name='aws-ml-blog-sagemaker-census-segmentation'
```

Take a look at the contents of this bucket; get a list of objects that are contained within the bucket and print out the names of the objects. You should see that there is one file, 'Census_Data_for_SageMaker.csv'.


```python
# get a list of objects in the bucket
obj_list=s3_client.list_objects(Bucket=bucket_name)

# print object(s)in S3 bucket
files=[]
for contents in obj_list['Contents']:
    files.append(contents['Key'])
    
print(files)
```

    ['Census_Data_for_SageMaker.csv']



```python
# there is one file --> one key
file_name=files[0]

print(file_name)
```

    Census_Data_for_SageMaker.csv


Retrieve the data file from the bucket with a call to `client.get_object()`.


```python
# get an S3 object by passing in the bucket and file name
data_object = s3_client.get_object(Bucket=bucket_name, Key=file_name)

# what info does the object contain?
display(data_object)
```


    {'ResponseMetadata': {'RequestId': 'DZHV0PNYQN254M08',
      'HostId': 'D1i8Hh6lgO1MDUW23sXJKsQ5JY9rhb9dCUy3F9tYoX7fbxv/3jqNTGuE+r7LOWgp4zd468zit6o=',
      'HTTPStatusCode': 200,
      'HTTPHeaders': {'x-amz-id-2': 'D1i8Hh6lgO1MDUW23sXJKsQ5JY9rhb9dCUy3F9tYoX7fbxv/3jqNTGuE+r7LOWgp4zd468zit6o=',
       'x-amz-request-id': 'DZHV0PNYQN254M08',
       'date': 'Sun, 16 May 2021 11:00:46 GMT',
       'last-modified': 'Wed, 12 Sep 2018 15:13:37 GMT',
       'etag': '"066d37f43f7762f1eb409b1660fe9763"',
       'accept-ranges': 'bytes',
       'content-type': 'text/csv',
       'content-length': '613237',
       'server': 'AmazonS3'},
      'RetryAttempts': 0},
     'AcceptRanges': 'bytes',
     'LastModified': datetime.datetime(2018, 9, 12, 15, 13, 37, tzinfo=tzutc()),
     'ContentLength': 613237,
     'ETag': '"066d37f43f7762f1eb409b1660fe9763"',
     'ContentType': 'text/csv',
     'Metadata': {},
     'Body': <botocore.response.StreamingBody at 0x7f25586bf390>}



```python
# information is in the "Body" of the object
data_body = data_object["Body"].read()
print('Data type: ', type(data_body))
```

    Data type:  <class 'bytes'>


This is a `bytes` datatype, which you can read it in using [io.BytesIO(file)](https://docs.python.org/3/library/io.html#binary-i-o).


```python
# read in bytes data
data_stream = io.BytesIO(data_body)

# create a dataframe
counties_df = pd.read_csv(data_stream, header=0, delimiter=",") 
counties_df.head()
```




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
      <th>CensusId</th>
      <th>State</th>
      <th>County</th>
      <th>TotalPop</th>
      <th>Men</th>
      <th>Women</th>
      <th>Hispanic</th>
      <th>White</th>
      <th>Black</th>
      <th>Native</th>
      <th>...</th>
      <th>Walk</th>
      <th>OtherTransp</th>
      <th>WorkAtHome</th>
      <th>MeanCommute</th>
      <th>Employed</th>
      <th>PrivateWork</th>
      <th>PublicWork</th>
      <th>SelfEmployed</th>
      <th>FamilyWork</th>
      <th>Unemployment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1001</td>
      <td>Alabama</td>
      <td>Autauga</td>
      <td>55221</td>
      <td>26745</td>
      <td>28476</td>
      <td>2.6</td>
      <td>75.8</td>
      <td>18.5</td>
      <td>0.4</td>
      <td>...</td>
      <td>0.5</td>
      <td>1.3</td>
      <td>1.8</td>
      <td>26.5</td>
      <td>23986</td>
      <td>73.6</td>
      <td>20.9</td>
      <td>5.5</td>
      <td>0.0</td>
      <td>7.6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1003</td>
      <td>Alabama</td>
      <td>Baldwin</td>
      <td>195121</td>
      <td>95314</td>
      <td>99807</td>
      <td>4.5</td>
      <td>83.1</td>
      <td>9.5</td>
      <td>0.6</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.4</td>
      <td>3.9</td>
      <td>26.4</td>
      <td>85953</td>
      <td>81.5</td>
      <td>12.3</td>
      <td>5.8</td>
      <td>0.4</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1005</td>
      <td>Alabama</td>
      <td>Barbour</td>
      <td>26932</td>
      <td>14497</td>
      <td>12435</td>
      <td>4.6</td>
      <td>46.2</td>
      <td>46.7</td>
      <td>0.2</td>
      <td>...</td>
      <td>1.8</td>
      <td>1.5</td>
      <td>1.6</td>
      <td>24.1</td>
      <td>8597</td>
      <td>71.8</td>
      <td>20.8</td>
      <td>7.3</td>
      <td>0.1</td>
      <td>17.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1007</td>
      <td>Alabama</td>
      <td>Bibb</td>
      <td>22604</td>
      <td>12073</td>
      <td>10531</td>
      <td>2.2</td>
      <td>74.5</td>
      <td>21.4</td>
      <td>0.4</td>
      <td>...</td>
      <td>0.6</td>
      <td>1.5</td>
      <td>0.7</td>
      <td>28.8</td>
      <td>8294</td>
      <td>76.8</td>
      <td>16.1</td>
      <td>6.7</td>
      <td>0.4</td>
      <td>8.3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1009</td>
      <td>Alabama</td>
      <td>Blount</td>
      <td>57710</td>
      <td>28512</td>
      <td>29198</td>
      <td>8.6</td>
      <td>87.9</td>
      <td>1.5</td>
      <td>0.3</td>
      <td>...</td>
      <td>0.9</td>
      <td>0.4</td>
      <td>2.3</td>
      <td>34.9</td>
      <td>22189</td>
      <td>82.0</td>
      <td>13.5</td>
      <td>4.2</td>
      <td>0.4</td>
      <td>7.7</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 37 columns</p>
</div>



## Exploratory Data Analysis (EDA)

Now that you've loaded in the data, it is time to clean it up, explore it, and pre-process it. Data exploration is one of the most important parts of the machine learning workflow because it allows you to notice any initial patterns in data distribution and features that may inform how you proceed with modeling and clustering the data.

### EXERCISE: Explore data & drop any incomplete rows of data

When you first explore the data, it is good to know what you are working with. How many data points and features are you starting with, and what kind of information can you get at a first glance? In this notebook, you're required to use complete data points to train a model. So, your first exercise will be to investigate the shape of this data and implement a simple, data cleaning step: dropping any incomplete rows of data.

You should be able to answer the **question**: How many data points and features are in the original, provided dataset? (And how many points are left after dropping any incomplete rows?)


```python
# print out stats about data
counties_df.shape
```




    (3220, 37)




```python
counties_df.isna().sum()[counties_df.isna().sum()>0]
```




    Income          1
    IncomeErr       1
    ChildPoverty    1
    dtype: int64




```python
# drop any incomplete rows of data, and create a new df
clean_counties_df = counties_df.dropna(axis=0)
```


```python
clean_counties_df.shape
```




    (3218, 37)



### EXERCISE: Create a new DataFrame, indexed by 'State-County'

Eventually, you'll want to feed these features into a machine learning model. Machine learning models need numerical data to learn from and not categorical data like strings (State, County). So, you'll reformat this data such that it is indexed by region and you'll also drop any features that are not useful for clustering.

To complete this task, perform the following steps, using your *clean* DataFrame, generated above:
1. Combine the descriptive columns, 'State' and 'County', into one, new categorical column, 'State-County'. 
2. Index the data by this unique State-County name.
3. After doing this, drop the old State and County columns and the CensusId column, which does not give us any meaningful demographic information.

After completing this task, you should have a DataFrame with 'State-County' as the index, and 34 columns of numerical data for each county. You should get a resultant DataFrame that looks like the following (truncated for display purposes):
```
                TotalPop	 Men	  Women	Hispanic	...
                
Alabama-Autauga	55221	 26745	28476	2.6         ...
Alabama-Baldwin	195121	95314	99807	4.5         ...
Alabama-Barbour	26932	 14497	12435	4.6         ...
...

```


```python
clean_counties_df.index=clean_counties_df['State'] + "-" + clean_counties_df['County']
```


```python
clean_counties_df.head()[:5]
```




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
      <th>CensusId</th>
      <th>State</th>
      <th>County</th>
      <th>TotalPop</th>
      <th>Men</th>
      <th>Women</th>
      <th>Hispanic</th>
      <th>White</th>
      <th>Black</th>
      <th>Native</th>
      <th>...</th>
      <th>Walk</th>
      <th>OtherTransp</th>
      <th>WorkAtHome</th>
      <th>MeanCommute</th>
      <th>Employed</th>
      <th>PrivateWork</th>
      <th>PublicWork</th>
      <th>SelfEmployed</th>
      <th>FamilyWork</th>
      <th>Unemployment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Alabama-Autauga</th>
      <td>1001</td>
      <td>Alabama</td>
      <td>Autauga</td>
      <td>55221</td>
      <td>26745</td>
      <td>28476</td>
      <td>2.6</td>
      <td>75.8</td>
      <td>18.5</td>
      <td>0.4</td>
      <td>...</td>
      <td>0.5</td>
      <td>1.3</td>
      <td>1.8</td>
      <td>26.5</td>
      <td>23986</td>
      <td>73.6</td>
      <td>20.9</td>
      <td>5.5</td>
      <td>0.0</td>
      <td>7.6</td>
    </tr>
    <tr>
      <th>Alabama-Baldwin</th>
      <td>1003</td>
      <td>Alabama</td>
      <td>Baldwin</td>
      <td>195121</td>
      <td>95314</td>
      <td>99807</td>
      <td>4.5</td>
      <td>83.1</td>
      <td>9.5</td>
      <td>0.6</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.4</td>
      <td>3.9</td>
      <td>26.4</td>
      <td>85953</td>
      <td>81.5</td>
      <td>12.3</td>
      <td>5.8</td>
      <td>0.4</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>Alabama-Barbour</th>
      <td>1005</td>
      <td>Alabama</td>
      <td>Barbour</td>
      <td>26932</td>
      <td>14497</td>
      <td>12435</td>
      <td>4.6</td>
      <td>46.2</td>
      <td>46.7</td>
      <td>0.2</td>
      <td>...</td>
      <td>1.8</td>
      <td>1.5</td>
      <td>1.6</td>
      <td>24.1</td>
      <td>8597</td>
      <td>71.8</td>
      <td>20.8</td>
      <td>7.3</td>
      <td>0.1</td>
      <td>17.6</td>
    </tr>
    <tr>
      <th>Alabama-Bibb</th>
      <td>1007</td>
      <td>Alabama</td>
      <td>Bibb</td>
      <td>22604</td>
      <td>12073</td>
      <td>10531</td>
      <td>2.2</td>
      <td>74.5</td>
      <td>21.4</td>
      <td>0.4</td>
      <td>...</td>
      <td>0.6</td>
      <td>1.5</td>
      <td>0.7</td>
      <td>28.8</td>
      <td>8294</td>
      <td>76.8</td>
      <td>16.1</td>
      <td>6.7</td>
      <td>0.4</td>
      <td>8.3</td>
    </tr>
    <tr>
      <th>Alabama-Blount</th>
      <td>1009</td>
      <td>Alabama</td>
      <td>Blount</td>
      <td>57710</td>
      <td>28512</td>
      <td>29198</td>
      <td>8.6</td>
      <td>87.9</td>
      <td>1.5</td>
      <td>0.3</td>
      <td>...</td>
      <td>0.9</td>
      <td>0.4</td>
      <td>2.3</td>
      <td>34.9</td>
      <td>22189</td>
      <td>82.0</td>
      <td>13.5</td>
      <td>4.2</td>
      <td>0.4</td>
      <td>7.7</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 37 columns</p>
</div>




```python
# drop the old State and County columns, and the CensusId column
# clean df should be modified or created anew
clean_counties_df = clean_counties_df.drop(['CensusId','State','County'], axis=1)

```


```python
clean_counties_df.head()[:3]
```




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
      <th>TotalPop</th>
      <th>Men</th>
      <th>Women</th>
      <th>Hispanic</th>
      <th>White</th>
      <th>Black</th>
      <th>Native</th>
      <th>Asian</th>
      <th>Pacific</th>
      <th>Citizen</th>
      <th>...</th>
      <th>Walk</th>
      <th>OtherTransp</th>
      <th>WorkAtHome</th>
      <th>MeanCommute</th>
      <th>Employed</th>
      <th>PrivateWork</th>
      <th>PublicWork</th>
      <th>SelfEmployed</th>
      <th>FamilyWork</th>
      <th>Unemployment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Alabama-Autauga</th>
      <td>55221</td>
      <td>26745</td>
      <td>28476</td>
      <td>2.6</td>
      <td>75.8</td>
      <td>18.5</td>
      <td>0.4</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>40725</td>
      <td>...</td>
      <td>0.5</td>
      <td>1.3</td>
      <td>1.8</td>
      <td>26.5</td>
      <td>23986</td>
      <td>73.6</td>
      <td>20.9</td>
      <td>5.5</td>
      <td>0.0</td>
      <td>7.6</td>
    </tr>
    <tr>
      <th>Alabama-Baldwin</th>
      <td>195121</td>
      <td>95314</td>
      <td>99807</td>
      <td>4.5</td>
      <td>83.1</td>
      <td>9.5</td>
      <td>0.6</td>
      <td>0.7</td>
      <td>0.0</td>
      <td>147695</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.4</td>
      <td>3.9</td>
      <td>26.4</td>
      <td>85953</td>
      <td>81.5</td>
      <td>12.3</td>
      <td>5.8</td>
      <td>0.4</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>Alabama-Barbour</th>
      <td>26932</td>
      <td>14497</td>
      <td>12435</td>
      <td>4.6</td>
      <td>46.2</td>
      <td>46.7</td>
      <td>0.2</td>
      <td>0.4</td>
      <td>0.0</td>
      <td>20714</td>
      <td>...</td>
      <td>1.8</td>
      <td>1.5</td>
      <td>1.6</td>
      <td>24.1</td>
      <td>8597</td>
      <td>71.8</td>
      <td>20.8</td>
      <td>7.3</td>
      <td>0.1</td>
      <td>17.6</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 34 columns</p>
</div>



Now, what features do you have to work with?


```python
# features
features_list = clean_counties_df.columns.values
print('Features: \n', features_list)
```

    Features: 
     ['TotalPop' 'Men' 'Women' 'Hispanic' 'White' 'Black' 'Native' 'Asian'
     'Pacific' 'Citizen' 'Income' 'IncomeErr' 'IncomePerCap' 'IncomePerCapErr'
     'Poverty' 'ChildPoverty' 'Professional' 'Service' 'Office' 'Construction'
     'Production' 'Drive' 'Carpool' 'Transit' 'Walk' 'OtherTransp'
     'WorkAtHome' 'MeanCommute' 'Employed' 'PrivateWork' 'PublicWork'
     'SelfEmployed' 'FamilyWork' 'Unemployment']


## Visualizing the Data

In general, you can see that features come in a variety of ranges, mostly percentages from 0-100, and counts that are integer values in a large range. Let's visualize the data in some of our feature columns and see what the distribution, over all counties, looks like.

The below cell displays **histograms**, which show the distribution of data points over discrete feature ranges. The x-axis represents the different bins; each bin is defined by a specific range of values that a feature can take, say between the values 0-5 and 5-10, and so on. The y-axis is the frequency of occurrence or the number of county data points that fall into each bin. I find it helpful to use the y-axis values for relative comparisons between different features.

Below, I'm plotting a histogram comparing methods of commuting to work over all of the counties. I just copied these feature names from the list of column names, printed above. I also know that all of these features are represented as percentages (%) in the original data, so the x-axes of these plots will be comparable.


```python
# transportation (to work)
transport_list = ['Drive', 'Carpool', 'Transit', 'Walk', 'OtherTransp']
n_bins = 30 # can decrease to get a wider bin (or vice versa)

for column_name in transport_list:
    ax=plt.subplots(figsize=(6,3))
    # get data by column_name and display a histogram
    ax = plt.hist(clean_counties_df[column_name], bins=n_bins)
    title="Histogram of " + column_name
    plt.title(title, fontsize=12)
    plt.show()
```


    
![png](output_27_0.png)
    



    
![png](output_27_1.png)
    



    
![png](output_27_2.png)
    



    
![png](output_27_3.png)
    



    
![png](output_27_4.png)
    


### EXERCISE: Create histograms of your own

Commute transportation method is just one category of features. If you take a look at the 34 features, you can see data on profession, race, income, and more. Display a set of histograms that interest you!



```python
# create a list of features that you want to compare or examine
my_list = ['WorkAtHome','Employed','PrivateWork','PublicWork','SelfEmployed','FamilyWork','Unemployment']
n_bins = 20 # define n_bins

# histogram creation code is similar to above
for column_name in my_list:
    ax=plt.subplots(figsize=(6,3))
    # get data by column_name and display a histogram
    ax = plt.hist(clean_counties_df[column_name], bins=n_bins)
    title="Histogram of " + column_name
    plt.title(title, fontsize=12)
    plt.show()
```


    
![png](output_29_0.png)
    



    
![png](output_29_1.png)
    



    
![png](output_29_2.png)
    



    
![png](output_29_3.png)
    



    
![png](output_29_4.png)
    



    
![png](output_29_5.png)
    



    
![png](output_29_6.png)
    


### EXERCISE: Normalize the data

You need to standardize the scale of the numerical columns in order to consistently compare the values of different features. You can use a [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) to transform the numerical values so that they all fall between 0 and 1.


```python
# scale numerical features into a normalized range, 0-1
# store them in this dataframe
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
counties_scaled = pd.DataFrame(scaler.fit_transform(clean_counties_df.astype(float)))

#get some features and state-county indices
counties_scaled.columns=clean_counties_df.columns
counties_scaled.index=clean_counties_df.index
counties_scaled.head()
```




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
      <th>TotalPop</th>
      <th>Men</th>
      <th>Women</th>
      <th>Hispanic</th>
      <th>White</th>
      <th>Black</th>
      <th>Native</th>
      <th>Asian</th>
      <th>Pacific</th>
      <th>Citizen</th>
      <th>...</th>
      <th>Walk</th>
      <th>OtherTransp</th>
      <th>WorkAtHome</th>
      <th>MeanCommute</th>
      <th>Employed</th>
      <th>PrivateWork</th>
      <th>PublicWork</th>
      <th>SelfEmployed</th>
      <th>FamilyWork</th>
      <th>Unemployment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Alabama-Autauga</th>
      <td>0.005475</td>
      <td>0.005381</td>
      <td>0.005566</td>
      <td>0.026026</td>
      <td>0.759519</td>
      <td>0.215367</td>
      <td>0.004343</td>
      <td>0.024038</td>
      <td>0.0</td>
      <td>0.006702</td>
      <td>...</td>
      <td>0.007022</td>
      <td>0.033248</td>
      <td>0.048387</td>
      <td>0.552430</td>
      <td>0.005139</td>
      <td>0.750000</td>
      <td>0.250000</td>
      <td>0.150273</td>
      <td>0.000000</td>
      <td>0.208219</td>
    </tr>
    <tr>
      <th>Alabama-Baldwin</th>
      <td>0.019411</td>
      <td>0.019246</td>
      <td>0.019572</td>
      <td>0.045045</td>
      <td>0.832665</td>
      <td>0.110594</td>
      <td>0.006515</td>
      <td>0.016827</td>
      <td>0.0</td>
      <td>0.024393</td>
      <td>...</td>
      <td>0.014045</td>
      <td>0.035806</td>
      <td>0.104839</td>
      <td>0.549872</td>
      <td>0.018507</td>
      <td>0.884354</td>
      <td>0.107616</td>
      <td>0.158470</td>
      <td>0.040816</td>
      <td>0.205479</td>
    </tr>
    <tr>
      <th>Alabama-Barbour</th>
      <td>0.002656</td>
      <td>0.002904</td>
      <td>0.002416</td>
      <td>0.046046</td>
      <td>0.462926</td>
      <td>0.543655</td>
      <td>0.002172</td>
      <td>0.009615</td>
      <td>0.0</td>
      <td>0.003393</td>
      <td>...</td>
      <td>0.025281</td>
      <td>0.038363</td>
      <td>0.043011</td>
      <td>0.491049</td>
      <td>0.001819</td>
      <td>0.719388</td>
      <td>0.248344</td>
      <td>0.199454</td>
      <td>0.010204</td>
      <td>0.482192</td>
    </tr>
    <tr>
      <th>Alabama-Bibb</th>
      <td>0.002225</td>
      <td>0.002414</td>
      <td>0.002042</td>
      <td>0.022022</td>
      <td>0.746493</td>
      <td>0.249127</td>
      <td>0.004343</td>
      <td>0.002404</td>
      <td>0.0</td>
      <td>0.002860</td>
      <td>...</td>
      <td>0.008427</td>
      <td>0.038363</td>
      <td>0.018817</td>
      <td>0.611253</td>
      <td>0.001754</td>
      <td>0.804422</td>
      <td>0.170530</td>
      <td>0.183060</td>
      <td>0.040816</td>
      <td>0.227397</td>
    </tr>
    <tr>
      <th>Alabama-Blount</th>
      <td>0.005722</td>
      <td>0.005738</td>
      <td>0.005707</td>
      <td>0.086086</td>
      <td>0.880762</td>
      <td>0.017462</td>
      <td>0.003257</td>
      <td>0.002404</td>
      <td>0.0</td>
      <td>0.006970</td>
      <td>...</td>
      <td>0.012640</td>
      <td>0.010230</td>
      <td>0.061828</td>
      <td>0.767263</td>
      <td>0.004751</td>
      <td>0.892857</td>
      <td>0.127483</td>
      <td>0.114754</td>
      <td>0.040816</td>
      <td>0.210959</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 34 columns</p>
</div>



---
# Data Modeling


Now, the data is ready to be fed into a machine learning model!

Each data point has 34 features, which means the data is 34-dimensional. Clustering algorithms rely on finding clusters in n-dimensional feature space. For higher dimensions, an algorithm like k-means has a difficult time figuring out which features are most important, and the result is, often, noisier clusters.

Some dimensions are not as important as others. For example, if every county in our dataset has the same rate of unemployment, then that particular feature doesn’t give us any distinguishing information; it will not help t separate counties into different groups because its value doesn’t *vary* between counties.

> Instead, we really want to find the features that help to separate and group data. We want to find features that cause the **most variance** in the dataset!

So, before I cluster this data, I’ll want to take a dimensionality reduction step. My aim will be to form a smaller set of features that will better help to separate our data. The technique I’ll use is called PCA or **principal component analysis**

## Dimensionality Reduction

PCA attempts to reduce the number of features within a dataset while retaining the “principal components”, which are defined as *weighted*, linear combinations of existing features that are designed to be linearly independent and account for the largest possible variability in the data! You can think of this method as taking many features and combining similar or redundant features together to form a new, smaller feature set.

We can reduce dimensionality with the built-in SageMaker model for PCA.

### Roles and Buckets

> To create a model, you'll first need to specify an IAM role, and to save the model attributes, you'll need to store them in an S3 bucket.

The `get_execution_role` function retrieves the IAM role you created at the time you created your notebook instance. Roles are essentially used to manage permissions and you can read more about that [in this documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html). For now, know that we have a FullAccess notebook, which allowed us to access and download the census data stored in S3.

You must specify a bucket name for an S3 bucket in your account where you want SageMaker model parameters to be stored. Note that the bucket must be in the same region as this notebook. You can get a default S3 bucket, which automatically creates a bucket for you and in your region, by storing the current SageMaker session and calling `session.default_bucket()`.


```python
from sagemaker import get_execution_role

session = sagemaker.Session() # store the current SageMaker session

# get IAM role
role = get_execution_role()
print(role)
```

    arn:aws:iam::061257208775:role/service-role/AmazonSageMaker-ExecutionRole-20210516T125114



```python
# get default bucket
bucket_name = session.default_bucket()
print(bucket_name)
print()
```

    sagemaker-us-east-1-061257208775
    


## Define a PCA Model

To create a PCA model, I'll use the built-in SageMaker resource. A SageMaker estimator requires a number of parameters to be specified; these define the type of training instance to use and the model hyperparameters. A PCA model requires the following constructor arguments:

* role: The IAM role, which was specified, above.
* train_instance_count: The number of training instances (typically, 1).
* train_instance_type: The type of SageMaker instance for training.
* num_components: An integer that defines the number of PCA components to produce.
* sagemaker_session: The session used to train on SageMaker.

Documentation on the PCA model can be found [here](http://sagemaker.readthedocs.io/en/latest/pca.html).

Below, I first specify where to save the model training data, the `output_path`.


```python
# define location to store model artifacts
prefix = 'counties'

output_path='s3://{}/{}/'.format(bucket_name, prefix)

print('Training artifacts will be uploaded to: {}'.format(output_path))
```

    Training artifacts will be uploaded to: s3://sagemaker-us-east-1-061257208775/counties/



```python
# define a PCA model
from sagemaker import PCA

# this is current features - 1
# you'll select only a portion of these to use, later
N_COMPONENTS=33

pca_SM = PCA(role=role,
             train_instance_count=1,
             train_instance_type='ml.c4.xlarge',
             output_path=output_path, # specified, above
             num_components=N_COMPONENTS, 
             sagemaker_session=session)

```

    train_instance_count has been renamed in sagemaker>=2.
    See: https://sagemaker.readthedocs.io/en/stable/v2.html for details.
    train_instance_type has been renamed in sagemaker>=2.
    See: https://sagemaker.readthedocs.io/en/stable/v2.html for details.


### Convert data into a RecordSet format

Next, prepare the data for a built-in model by converting the DataFrame to a numpy array of float values.

The *record_set* function in the SageMaker PCA model converts a numpy array into a **RecordSet** format that is the required format for the training input data. This is a requirement for _all_ of SageMaker's built-in models. The use of this data type is one of the reasons that allows training of models within Amazon SageMaker to perform faster, especially for large datasets.


```python
# convert df to np array
train_data_np = counties_scaled.values.astype('float32')

# convert to RecordSet format
formatted_train_data = pca_SM.record_set(train_data_np)
```

## Train the model

Call the fit function on the PCA model, passing in our formatted, training data. This spins up a training instance to perform the training job.

Note that it takes the longest to launch the specified training instance; the fitting itself doesn't take much time.


```python
%%time

# train the PCA mode on the formatted data
pca_SM.fit(formatted_train_data)
```

    Defaulting to the only supported framework/algorithm version: 1. Ignoring framework/algorithm version: 1.
    Defaulting to the only supported framework/algorithm version: 1. Ignoring framework/algorithm version: 1.


    2021-05-16 11:52:21 Starting - Starting the training job...
    2021-05-16 11:52:22 Starting - Launching requested ML instancesProfilerReport-1621165940: InProgress
    ......
    2021-05-16 11:53:35 Starting - Preparing the instances for training.........
    2021-05-16 11:55:16 Downloading - Downloading input data
    2021-05-16 11:55:16 Training - Downloading the training image...
    2021-05-16 11:55:49 Uploading - Uploading generated training model
    2021-05-16 11:55:49 Completed - Training job completed
    [34mDocker entrypoint called with argument(s): train[0m
    [34mRunning default environment configuration script[0m
    [34m[05/16/2021 11:55:40 INFO 139872936527680] Reading default configuration from /opt/amazon/lib/python3.7/site-packages/algorithm/resources/default-conf.json: {'algorithm_mode': 'regular', 'subtract_mean': 'true', 'extra_components': '-1', 'force_dense': 'true', 'epochs': 1, '_log_level': 'info', '_kvstore': 'dist_sync', '_num_kv_servers': 'auto', '_num_gpus': 'auto'}[0m
    [34m[05/16/2021 11:55:40 INFO 139872936527680] Merging with provided configuration from /opt/ml/input/config/hyperparameters.json: {'feature_dim': '34', 'num_components': '33', 'mini_batch_size': '500'}[0m
    [34m[05/16/2021 11:55:40 INFO 139872936527680] Final configuration: {'algorithm_mode': 'regular', 'subtract_mean': 'true', 'extra_components': '-1', 'force_dense': 'true', 'epochs': 1, '_log_level': 'info', '_kvstore': 'dist_sync', '_num_kv_servers': 'auto', '_num_gpus': 'auto', 'feature_dim': '34', 'num_components': '33', 'mini_batch_size': '500'}[0m
    [34m[05/16/2021 11:55:40 WARNING 139872936527680] Loggers have already been setup.[0m
    [34m[05/16/2021 11:55:40 INFO 139872936527680] Launching parameter server for role scheduler[0m
    [34m[05/16/2021 11:55:40 INFO 139872936527680] {'ENVROOT': '/opt/amazon', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'cpp', 'HOSTNAME': 'ip-10-2-190-141.ec2.internal', 'TRAINING_JOB_NAME': 'pca-2021-05-16-11-52-20-920', 'NVIDIA_REQUIRE_CUDA': 'cuda>=9.0', 'TRAINING_JOB_ARN': 'arn:aws:sagemaker:us-east-1:061257208775:training-job/pca-2021-05-16-11-52-20-920', 'AWS_CONTAINER_CREDENTIALS_RELATIVE_URI': '/v2/credentials/bbd40ea8-1fa7-4628-8eaf-4eb08d09cffa', 'CANONICAL_ENVROOT': '/opt/amazon', 'PYTHONUNBUFFERED': 'TRUE', 'NVIDIA_VISIBLE_DEVICES': 'void', 'LD_LIBRARY_PATH': '/opt/amazon/lib/python3.7/site-packages/cv2/../../../../lib:/usr/local/nvidia/lib64:/opt/amazon/lib', 'NVIDIA_DRIVER_CAPABILITIES': 'compute,utility', 'AWS_EXECUTION_ENV': 'AWS_ECS_EC2', 'PATH': '/opt/amazon/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/amazon/bin:/opt/amazon/bin', 'PWD': '/', 'LANG': 'en_US.utf8', 'AWS_REGION': 'us-east-1', 'HOME': '/root', 'SHLVL': '1', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION': '2', 'OMP_NUM_THREADS': '2', 'ECS_CONTAINER_METADATA_URI': 'http://169.254.170.2/v3/3191a72d-e08b-4f4d-86d0-1ce1e85ebb99', 'DMLC_INTERFACE': 'eth0', 'ECS_CONTAINER_METADATA_URI_V4': 'http://169.254.170.2/v4/3191a72d-e08b-4f4d-86d0-1ce1e85ebb99', 'SAGEMAKER_HTTP_PORT': '8080', 'SAGEMAKER_DATA_PATH': '/opt/ml'}[0m
    [34m[05/16/2021 11:55:40 INFO 139872936527680] envs={'ENVROOT': '/opt/amazon', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'cpp', 'HOSTNAME': 'ip-10-2-190-141.ec2.internal', 'TRAINING_JOB_NAME': 'pca-2021-05-16-11-52-20-920', 'NVIDIA_REQUIRE_CUDA': 'cuda>=9.0', 'TRAINING_JOB_ARN': 'arn:aws:sagemaker:us-east-1:061257208775:training-job/pca-2021-05-16-11-52-20-920', 'AWS_CONTAINER_CREDENTIALS_RELATIVE_URI': '/v2/credentials/bbd40ea8-1fa7-4628-8eaf-4eb08d09cffa', 'CANONICAL_ENVROOT': '/opt/amazon', 'PYTHONUNBUFFERED': 'TRUE', 'NVIDIA_VISIBLE_DEVICES': 'void', 'LD_LIBRARY_PATH': '/opt/amazon/lib/python3.7/site-packages/cv2/../../../../lib:/usr/local/nvidia/lib64:/opt/amazon/lib', 'NVIDIA_DRIVER_CAPABILITIES': 'compute,utility', 'AWS_EXECUTION_ENV': 'AWS_ECS_EC2', 'PATH': '/opt/amazon/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/amazon/bin:/opt/amazon/bin', 'PWD': '/', 'LANG': 'en_US.utf8', 'AWS_REGION': 'us-east-1', 'HOME': '/root', 'SHLVL': '1', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION': '2', 'OMP_NUM_THREADS': '2', 'ECS_CONTAINER_METADATA_URI': 'http://169.254.170.2/v3/3191a72d-e08b-4f4d-86d0-1ce1e85ebb99', 'DMLC_INTERFACE': 'eth0', 'ECS_CONTAINER_METADATA_URI_V4': 'http://169.254.170.2/v4/3191a72d-e08b-4f4d-86d0-1ce1e85ebb99', 'SAGEMAKER_HTTP_PORT': '8080', 'SAGEMAKER_DATA_PATH': '/opt/ml', 'DMLC_ROLE': 'scheduler', 'DMLC_PS_ROOT_URI': '10.2.190.141', 'DMLC_PS_ROOT_PORT': '9000', 'DMLC_NUM_SERVER': '1', 'DMLC_NUM_WORKER': '1'}[0m
    [34m[05/16/2021 11:55:40 INFO 139872936527680] Launching parameter server for role server[0m
    [34m[05/16/2021 11:55:40 INFO 139872936527680] {'ENVROOT': '/opt/amazon', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'cpp', 'HOSTNAME': 'ip-10-2-190-141.ec2.internal', 'TRAINING_JOB_NAME': 'pca-2021-05-16-11-52-20-920', 'NVIDIA_REQUIRE_CUDA': 'cuda>=9.0', 'TRAINING_JOB_ARN': 'arn:aws:sagemaker:us-east-1:061257208775:training-job/pca-2021-05-16-11-52-20-920', 'AWS_CONTAINER_CREDENTIALS_RELATIVE_URI': '/v2/credentials/bbd40ea8-1fa7-4628-8eaf-4eb08d09cffa', 'CANONICAL_ENVROOT': '/opt/amazon', 'PYTHONUNBUFFERED': 'TRUE', 'NVIDIA_VISIBLE_DEVICES': 'void', 'LD_LIBRARY_PATH': '/opt/amazon/lib/python3.7/site-packages/cv2/../../../../lib:/usr/local/nvidia/lib64:/opt/amazon/lib', 'NVIDIA_DRIVER_CAPABILITIES': 'compute,utility', 'AWS_EXECUTION_ENV': 'AWS_ECS_EC2', 'PATH': '/opt/amazon/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/amazon/bin:/opt/amazon/bin', 'PWD': '/', 'LANG': 'en_US.utf8', 'AWS_REGION': 'us-east-1', 'HOME': '/root', 'SHLVL': '1', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION': '2', 'OMP_NUM_THREADS': '2', 'ECS_CONTAINER_METADATA_URI': 'http://169.254.170.2/v3/3191a72d-e08b-4f4d-86d0-1ce1e85ebb99', 'DMLC_INTERFACE': 'eth0', 'ECS_CONTAINER_METADATA_URI_V4': 'http://169.254.170.2/v4/3191a72d-e08b-4f4d-86d0-1ce1e85ebb99', 'SAGEMAKER_HTTP_PORT': '8080', 'SAGEMAKER_DATA_PATH': '/opt/ml'}[0m
    [34m[05/16/2021 11:55:40 INFO 139872936527680] envs={'ENVROOT': '/opt/amazon', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'cpp', 'HOSTNAME': 'ip-10-2-190-141.ec2.internal', 'TRAINING_JOB_NAME': 'pca-2021-05-16-11-52-20-920', 'NVIDIA_REQUIRE_CUDA': 'cuda>=9.0', 'TRAINING_JOB_ARN': 'arn:aws:sagemaker:us-east-1:061257208775:training-job/pca-2021-05-16-11-52-20-920', 'AWS_CONTAINER_CREDENTIALS_RELATIVE_URI': '/v2/credentials/bbd40ea8-1fa7-4628-8eaf-4eb08d09cffa', 'CANONICAL_ENVROOT': '/opt/amazon', 'PYTHONUNBUFFERED': 'TRUE', 'NVIDIA_VISIBLE_DEVICES': 'void', 'LD_LIBRARY_PATH': '/opt/amazon/lib/python3.7/site-packages/cv2/../../../../lib:/usr/local/nvidia/lib64:/opt/amazon/lib', 'NVIDIA_DRIVER_CAPABILITIES': 'compute,utility', 'AWS_EXECUTION_ENV': 'AWS_ECS_EC2', 'PATH': '/opt/amazon/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/amazon/bin:/opt/amazon/bin', 'PWD': '/', 'LANG': 'en_US.utf8', 'AWS_REGION': 'us-east-1', 'HOME': '/root', 'SHLVL': '1', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION': '2', 'OMP_NUM_THREADS': '2', 'ECS_CONTAINER_METADATA_URI': 'http://169.254.170.2/v3/3191a72d-e08b-4f4d-86d0-1ce1e85ebb99', 'DMLC_INTERFACE': 'eth0', 'ECS_CONTAINER_METADATA_URI_V4': 'http://169.254.170.2/v4/3191a72d-e08b-4f4d-86d0-1ce1e85ebb99', 'SAGEMAKER_HTTP_PORT': '8080', 'SAGEMAKER_DATA_PATH': '/opt/ml', 'DMLC_ROLE': 'server', 'DMLC_PS_ROOT_URI': '10.2.190.141', 'DMLC_PS_ROOT_PORT': '9000', 'DMLC_NUM_SERVER': '1', 'DMLC_NUM_WORKER': '1'}[0m
    [34m[05/16/2021 11:55:40 INFO 139872936527680] Environment: {'ENVROOT': '/opt/amazon', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'cpp', 'HOSTNAME': 'ip-10-2-190-141.ec2.internal', 'TRAINING_JOB_NAME': 'pca-2021-05-16-11-52-20-920', 'NVIDIA_REQUIRE_CUDA': 'cuda>=9.0', 'TRAINING_JOB_ARN': 'arn:aws:sagemaker:us-east-1:061257208775:training-job/pca-2021-05-16-11-52-20-920', 'AWS_CONTAINER_CREDENTIALS_RELATIVE_URI': '/v2/credentials/bbd40ea8-1fa7-4628-8eaf-4eb08d09cffa', 'CANONICAL_ENVROOT': '/opt/amazon', 'PYTHONUNBUFFERED': 'TRUE', 'NVIDIA_VISIBLE_DEVICES': 'void', 'LD_LIBRARY_PATH': '/opt/amazon/lib/python3.7/site-packages/cv2/../../../../lib:/usr/local/nvidia/lib64:/opt/amazon/lib', 'NVIDIA_DRIVER_CAPABILITIES': 'compute,utility', 'AWS_EXECUTION_ENV': 'AWS_ECS_EC2', 'PATH': '/opt/amazon/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/amazon/bin:/opt/amazon/bin', 'PWD': '/', 'LANG': 'en_US.utf8', 'AWS_REGION': 'us-east-1', 'HOME': '/root', 'SHLVL': '1', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION': '2', 'OMP_NUM_THREADS': '2', 'ECS_CONTAINER_METADATA_URI': 'http://169.254.170.2/v3/3191a72d-e08b-4f4d-86d0-1ce1e85ebb99', 'DMLC_INTERFACE': 'eth0', 'ECS_CONTAINER_METADATA_URI_V4': 'http://169.254.170.2/v4/3191a72d-e08b-4f4d-86d0-1ce1e85ebb99', 'SAGEMAKER_HTTP_PORT': '8080', 'SAGEMAKER_DATA_PATH': '/opt/ml', 'DMLC_ROLE': 'worker', 'DMLC_PS_ROOT_URI': '10.2.190.141', 'DMLC_PS_ROOT_PORT': '9000', 'DMLC_NUM_SERVER': '1', 'DMLC_NUM_WORKER': '1'}[0m
    [34mProcess 34 is a shell:scheduler.[0m
    [34mProcess 43 is a shell:server.[0m
    [34mProcess 1 is a worker.[0m
    [34m[05/16/2021 11:55:40 INFO 139872936527680] Using default worker.[0m
    [34m[05/16/2021 11:55:40 INFO 139872936527680] Loaded iterator creator application/x-labeled-vector-protobuf for content type ('application/x-labeled-vector-protobuf', '1.0')[0m
    [34m[05/16/2021 11:55:40 INFO 139872936527680] Loaded iterator creator application/x-recordio-protobuf for content type ('application/x-recordio-protobuf', '1.0')[0m
    [34m[05/16/2021 11:55:40 INFO 139872936527680] Loaded iterator creator protobuf for content type ('protobuf', '1.0')[0m
    [34m[05/16/2021 11:55:40 INFO 139872936527680] Checkpoint loading and saving are disabled.[0m
    [34m[05/16/2021 11:55:40 INFO 139872936527680] Create Store: dist_sync[0m
    [34m[05/16/2021 11:55:41 INFO 139872936527680] nvidia-smi: took 0.029 seconds to run.[0m
    [34m[05/16/2021 11:55:41 INFO 139872936527680] nvidia-smi identified 0 GPUs.[0m
    [34m[05/16/2021 11:55:41 INFO 139872936527680] Number of GPUs being used: 0[0m
    [34m[05/16/2021 11:55:41 INFO 139872936527680] The default executor is <PCAExecutor on cpu(0)>.[0m
    [34m[05/16/2021 11:55:41 INFO 139872936527680] 34 feature(s) found in 'data'.[0m
    [34m[05/16/2021 11:55:41 INFO 139872936527680] <PCAExecutor on cpu(0)> is assigned to batch slice from 0 to 499.[0m
    [34m#metrics {"StartTime": 1621166140.3517475, "EndTime": 1621166141.1486742, "Dimensions": {"Algorithm": "PCA", "Host": "algo-1", "Operation": "training"}, "Metrics": {"initialize.time": {"sum": 760.80322265625, "count": 1, "min": 760.80322265625, "max": 760.80322265625}}}
    [0m
    [34m#metrics {"StartTime": 1621166141.1488686, "EndTime": 1621166141.1489124, "Dimensions": {"Algorithm": "PCA", "Host": "algo-1", "Operation": "training", "Meta": "init_train_data_iter"}, "Metrics": {"Total Records Seen": {"sum": 0.0, "count": 1, "min": 0, "max": 0}, "Total Batches Seen": {"sum": 0.0, "count": 1, "min": 0, "max": 0}, "Max Records Seen Between Resets": {"sum": 0.0, "count": 1, "min": 0, "max": 0}, "Max Batches Seen Between Resets": {"sum": 0.0, "count": 1, "min": 0, "max": 0}, "Reset Count": {"sum": 0.0, "count": 1, "min": 0, "max": 0}, "Number of Records Since Last Reset": {"sum": 0.0, "count": 1, "min": 0, "max": 0}, "Number of Batches Since Last Reset": {"sum": 0.0, "count": 1, "min": 0, "max": 0}}}
    [0m
    [34m[2021-05-16 11:55:41.149] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/train", "epoch": 0, "duration": 797, "num_examples": 1, "num_bytes": 82000}[0m
    [34m[2021-05-16 11:55:41.201] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/train", "epoch": 1, "duration": 43, "num_examples": 7, "num_bytes": 527752}[0m
    [34m#metrics {"StartTime": 1621166141.1488054, "EndTime": 1621166141.202244, "Dimensions": {"Algorithm": "PCA", "Host": "algo-1", "Operation": "training"}, "Metrics": {"epochs": {"sum": 1.0, "count": 1, "min": 1, "max": 1}, "update.time": {"sum": 52.892446517944336, "count": 1, "min": 52.892446517944336, "max": 52.892446517944336}}}
    [0m
    [34m[05/16/2021 11:55:41 INFO 139872936527680] #progress_metric: host=algo-1, completed 100.0 % of epochs[0m
    [34m#metrics {"StartTime": 1621166141.149317, "EndTime": 1621166141.2026322, "Dimensions": {"Algorithm": "PCA", "Host": "algo-1", "Operation": "training", "epoch": 0, "Meta": "training_data_iter"}, "Metrics": {"Total Records Seen": {"sum": 3218.0, "count": 1, "min": 3218, "max": 3218}, "Total Batches Seen": {"sum": 7.0, "count": 1, "min": 7, "max": 7}, "Max Records Seen Between Resets": {"sum": 3218.0, "count": 1, "min": 3218, "max": 3218}, "Max Batches Seen Between Resets": {"sum": 7.0, "count": 1, "min": 7, "max": 7}, "Reset Count": {"sum": 1.0, "count": 1, "min": 1, "max": 1}, "Number of Records Since Last Reset": {"sum": 3218.0, "count": 1, "min": 3218, "max": 3218}, "Number of Batches Since Last Reset": {"sum": 7.0, "count": 1, "min": 7, "max": 7}}}
    [0m
    [34m[05/16/2021 11:55:41 INFO 139872936527680] #throughput_metric: host=algo-1, train throughput=60188.22779831528 records/second[0m
    [34m#metrics {"StartTime": 1621166141.2023177, "EndTime": 1621166141.2292566, "Dimensions": {"Algorithm": "PCA", "Host": "algo-1", "Operation": "training"}, "Metrics": {"finalize.time": {"sum": 26.148080825805664, "count": 1, "min": 26.148080825805664, "max": 26.148080825805664}}}
    [0m
    [34m[05/16/2021 11:55:41 INFO 139872936527680] Test data is not provided.[0m
    [34m#metrics {"StartTime": 1621166141.229343, "EndTime": 1621166141.230895, "Dimensions": {"Algorithm": "PCA", "Host": "algo-1", "Operation": "training"}, "Metrics": {"setuptime": {"sum": 24.92070198059082, "count": 1, "min": 24.92070198059082, "max": 24.92070198059082}, "totaltime": {"sum": 1043.4308052062988, "count": 1, "min": 1043.4308052062988, "max": 1043.4308052062988}}}
    [0m
    Training seconds: 41
    Billable seconds: 41
    CPU times: user 495 ms, sys: 22.4 ms, total: 517 ms
    Wall time: 3min 42s


## Accessing the PCA Model Attributes

After the model is trained, we can access the underlying model parameters.

### Unzip the Model Details

Now that the training job is complete, you can find the job under **Jobs** in the **Training**  subsection  in the Amazon SageMaker console. You can find the job name listed in the training jobs. Use that job name in the following code to specify which model to examine.

Model artifacts are stored in S3 as a TAR file; a compressed file in the output path we specified + 'output/model.tar.gz'. The artifacts stored here can be used to deploy a trained model.


```python
# Get the name of the training job, it's suggested that you copy-paste
# from the notebook or from a specific job in the AWS console

training_job_name='pca-2021-05-16-11-52-20-920'

# where the model is saved, by default
model_key = os.path.join(prefix, training_job_name, 'output/model.tar.gz')
print(model_key)

# download and unzip model
boto3.resource('s3').Bucket(bucket_name).download_file(model_key, 'model.tar.gz')

# unzipping as model_algo-1
os.system('tar -zxvf model.tar.gz')
os.system('unzip model_algo-1')
```

    counties/pca-2021-05-16-11-52-20-920/output/model.tar.gz





    2304



### MXNet Array

Many of the Amazon SageMaker algorithms use MXNet for computational speed, including PCA, and so the model artifacts are stored as an array. After the model is unzipped and decompressed, we can load the array using MXNet.

You can take a look at the MXNet [documentation, here](https://aws.amazon.com/mxnet/).


```python
import mxnet as mx

# loading the unzipped artifacts
pca_model_params = mx.ndarray.load('model_algo-1')

# what are the params
print(pca_model_params)
```

    {'s': 
    [1.7896362e-02 3.0864021e-02 3.2130770e-02 3.5486195e-02 9.4831578e-02
     1.2699370e-01 4.0288666e-01 1.4084760e+00 1.5100485e+00 1.5957943e+00
     1.7783760e+00 2.1662524e+00 2.2966361e+00 2.3856051e+00 2.6954880e+00
     2.8067985e+00 3.0175958e+00 3.3952675e+00 3.5731301e+00 3.6966958e+00
     4.1890211e+00 4.3457499e+00 4.5410376e+00 5.0189657e+00 5.5786467e+00
     5.9809699e+00 6.3925138e+00 7.6952214e+00 7.9913125e+00 1.0180052e+01
     1.1718245e+01 1.3035975e+01 1.9592180e+01]
    <NDArray 33 @cpu(0)>, 'v': 
    [[ 2.46869749e-03  2.56468095e-02  2.50773830e-03 ... -7.63925165e-02
       1.59879066e-02  5.04589686e-03]
     [-2.80601848e-02 -6.86634064e-01 -1.96283013e-02 ... -7.59587288e-02
       1.57304872e-02  4.95312130e-03]
     [ 3.25766727e-02  7.17300594e-01  2.40726061e-02 ... -7.68136829e-02
       1.62378680e-02  5.13597298e-03]
     ...
     [ 1.12151138e-01 -1.17030945e-02 -2.88011521e-01 ...  1.39890045e-01
      -3.09406728e-01 -6.34506866e-02]
     [ 2.99992133e-02 -3.13433539e-03 -7.63589665e-02 ...  4.17341813e-02
      -7.06735924e-02 -1.42857227e-02]
     [ 7.33537527e-05  3.01008171e-04 -8.00925500e-06 ...  6.97060227e-02
       1.20169498e-01  2.33626723e-01]]
    <NDArray 34x33 @cpu(0)>, 'mean': 
    [[0.00988273 0.00986636 0.00989863 0.11017046 0.7560245  0.10094159
      0.0186819  0.02940491 0.0064698  0.01154038 0.31539047 0.1222766
      0.3030056  0.08220861 0.256217   0.2964254  0.28914267 0.40191284
      0.57868284 0.2854676  0.28294644 0.82774544 0.34378946 0.01576072
      0.04649627 0.04115358 0.12442778 0.47014    0.00980645 0.7608103
      0.19442631 0.21674445 0.0294168  0.22177474]]
    <NDArray 1x34 @cpu(0)>}


## PCA Model Attributes

Three types of model attributes are contained within the PCA model.

* **mean**: The mean that was subtracted from a component in order to center it.
* **v**: The makeup of the principal components; (same as ‘components_’ in an sklearn PCA model).
* **s**: The singular values of the components for the PCA transformation. This does not exactly give the % variance from the original feature space, but can give the % variance from the projected feature space.
    
We are only interested in v and s. 

From s, we can get an approximation of the data variance that is covered in the first `n` principal components. The approximate explained variance is given by the formula: the sum of squared s values for all top n components over the sum over squared s values for _all_ components:

\begin{equation*}
\frac{\sum_{n}^{ } s_n^2}{\sum s^2}
\end{equation*}

From v, we can learn more about the combinations of original features that make up each principal component.



```python
# get selected params
s=pd.DataFrame(pca_model_params['s'].asnumpy())
v=pd.DataFrame(pca_model_params['v'].asnumpy())
```

## Data Variance

Our current PCA model creates 33 principal components, but when we create new dimensionality-reduced training data, we'll only select a few, top n components to use. To decide how many top components to include, it's helpful to look at how much **data variance** the components capture. For our original, high-dimensional data, 34 features captured 100% of our data variance. If we discard some of these higher dimensions, we will lower the amount of variance we can capture.

### Tradeoff: dimensionality vs. data variance

As an illustrative example, say we have original data in three dimensions. So, three dimensions capture 100% of our data variance; these dimensions cover the entire spread of our data. The below images are taken from the PhD thesis,  [“Approaches to analyse and interpret biological profile data”](https://publishup.uni-potsdam.de/opus4-ubp/frontdoor/index/index/docId/696) by Matthias Scholz, (2006, University of Potsdam, Germany).

<img src='notebook_ims/3d_original_data.png' width=35% />

Now, you may also note that most of this data seems related; it falls close to a 2D plane, and just by looking at the spread of the data, we  can visualize that the original, three dimensions have some correlation. So, we can instead choose to create two new dimensions, made up of linear combinations of the original, three dimensions. These dimensions are represented by the two axes/lines, centered in the data. 

<img src='notebook_ims/pca_2d_dim_reduction.png' width=70% />

If we project this in a new, 2D space, we can see that we still capture most of the original data variance using *just* two dimensions. There is a tradeoff between the amount of variance we can capture and the number of component-dimensions we use to represent our data.

When we select the top n components to use in a new data model, we'll typically want to include enough components to capture about 80-90% of the original data variance. In this project, we are looking at generalizing over a lot of data and we'll aim for about 80% coverage.

**Note**: The _top_ principal components, with the largest s values, are actually at the end of the s DataFrame. Let's print out the s values for the top n, principal components.


```python
# looking at top 5 components
n_principal_components = 5

start_idx = N_COMPONENTS - n_principal_components  # 33-n

# print a selection of s
print(s.iloc[start_idx:, :])
```

                0
    28   7.991313
    29  10.180052
    30  11.718245
    31  13.035975
    32  19.592180


### EXERCISE: Calculate the explained variance

In creating new training data, you'll want to choose the top n principal components that account for at least 80% data variance. 

Complete a function, `explained_variance` that takes in the entire array `s` and a number of top principal components to consider. Then return the approximate, explained variance for those top n components. 

For example, to calculate the explained variance for the top 5 components, calculate s squared for *each* of the top 5 components, add those up and normalize by the sum of *all* squared s values, according to this formula:

\begin{equation*}
\frac{\sum_{5}^{ } s_n^2}{\sum s^2}
\end{equation*}

> Using this function, you should be able to answer the **question**: What is the smallest number of principal components that captures at least 80% of the total variance in the dataset?


```python
# Calculate the explained variance for the top n principal components
# you may assume you have access to the global var N_COMPONENTS
def explained_variance(s, n_top_components):
    '''Calculates the approx. data variance that n_top_components captures.
       :param s: A dataframe of singular values for top components; 
           the top value is in the last row.
       :param n_top_components: An integer, the number of top components to use.
       :return: The expected data variance covered by the n_top_components.'''
    
    start_idx = N_COMPONENTS - n_top_components  ## 33-3 = 30, for example
    # calculate approx variance
    exp_variance = np.square(s.iloc[start_idx:,:]).sum()/np.square(s).sum()
    
    return exp_variance[0]
```

### Test Cell

Test out your own code by seeing how it responds to different inputs; does it return a reasonable value for the single, top component? What about for the top 5 components?


```python
# test cell
n_top_components = 7 # select a value for the number of top components

# calculate the explained variance
exp_variance = explained_variance(s, n_top_components)
print('Explained variance: ', exp_variance)
```

    Explained variance:  0.80167246


As an example, you should see that the top principal component accounts for about 32% of our data variance! Next, you may be wondering what makes up this (and other components); what linear combination of features make these components so influential in describing the spread of our data?

Below, let's take a look at our original features and use that as a reference.


```python
# features
features_list = counties_scaled.columns.values
print('Features: \n', features_list)
```

    Features: 
     ['TotalPop' 'Men' 'Women' 'Hispanic' 'White' 'Black' 'Native' 'Asian'
     'Pacific' 'Citizen' 'Income' 'IncomeErr' 'IncomePerCap' 'IncomePerCapErr'
     'Poverty' 'ChildPoverty' 'Professional' 'Service' 'Office' 'Construction'
     'Production' 'Drive' 'Carpool' 'Transit' 'Walk' 'OtherTransp'
     'WorkAtHome' 'MeanCommute' 'Employed' 'PrivateWork' 'PublicWork'
     'SelfEmployed' 'FamilyWork' 'Unemployment']


## Component Makeup

We can now examine the makeup of each PCA component based on **the weightings of the original features that are included in the component**. The following code shows the feature-level makeup of the first component.

Note that the components are again ordered from smallest to largest and so I am getting the correct rows by calling N_COMPONENTS-1 to get the top, 1, component.


```python
import seaborn as sns

def display_component(v, features_list, component_num, n_weights=10):
    
    # get index of component (last row - component_num)
    row_idx = N_COMPONENTS-component_num

    # get the list of weights from a row in v, dataframe
    v_1_row = v.iloc[:, row_idx]
    v_1 = np.squeeze(v_1_row.values)

    # match weights to features in counties_scaled dataframe, using list comporehension
    comps = pd.DataFrame(list(zip(v_1, features_list)), 
                         columns=['weights', 'features'])

    # we'll want to sort by the largest n_weights
    # weights can be neg/pos and we'll sort by magnitude
    comps['abs_weights']=comps['weights'].apply(lambda x: np.abs(x))
    sorted_weight_data = comps.sort_values('abs_weights', ascending=False).head(n_weights)

    # display using seaborn
    ax=plt.subplots(figsize=(10,6))
    ax=sns.barplot(data=sorted_weight_data, 
                   x="weights", 
                   y="features", 
                   palette="Blues_d")
    ax.set_title("PCA Component Makeup, Component #" + str(component_num))
    plt.show()

```


```python
# display makeup of first component
num=7
display_component(v, counties_scaled.columns.values, component_num=num, n_weights=12)
```


    
![png](output_60_0.png)
    


# Deploying the PCA Model

We can now deploy this model and use it to make "predictions". Instead of seeing what happens with some test data, we'll actually want to pass our training data into the deployed endpoint to create principal components for each data point. 

Run the cell below to deploy/host this model on an instance_type that we specify.


```python
%%time
# this takes a little while, around 7mins
pca_predictor = pca_SM.deploy(initial_instance_count=1, 
                              instance_type='ml.t2.medium')
```

    Defaulting to the only supported framework/algorithm version: 1. Ignoring framework/algorithm version: 1.


    -------------------!CPU times: user 346 ms, sys: 18 ms, total: 364 ms
    Wall time: 9min 33s


We can pass the original, numpy dataset to the model and transform the data using the model we created. Then we can take the largest n components to reduce the dimensionality of our data.


```python
# pass np train data to the PCA model
train_pca = pca_predictor.predict(train_data_np)
```


```python
# check out the first item in the produced training features
data_idx = 0
print(train_pca[data_idx])
```

    label {
      key: "projection"
      value {
        float32_tensor {
          values: 0.0002009272575378418
          values: 0.0002455431967973709
          values: -0.0005782842636108398
          values: -0.0007815659046173096
          values: -0.00041911262087523937
          values: -0.0005133943632245064
          values: -0.0011316537857055664
          values: 0.0017268601804971695
          values: -0.005361668765544891
          values: -0.009066537022590637
          values: -0.008141040802001953
          values: -0.004735097289085388
          values: -0.00716288760304451
          values: 0.0003725700080394745
          values: -0.01208949089050293
          values: 0.02134685218334198
          values: 0.0009293854236602783
          values: 0.002417147159576416
          values: -0.0034637749195098877
          values: 0.01794189214706421
          values: -0.01639425754547119
          values: 0.06260128319263458
          values: 0.06637358665466309
          values: 0.002479255199432373
          values: 0.10011336207389832
          values: -0.1136140376329422
          values: 0.02589476853609085
          values: 0.04045158624649048
          values: -0.01082391943782568
          values: 0.1204797774553299
          values: -0.0883558839559555
          values: 0.16052711009979248
          values: -0.06027412414550781
        }
      }
    }
    


### EXERCISE: Create a transformed DataFrame

For each of our data points, get the top n component values from the list of component data points, returned by our predictor above, and put those into a new DataFrame.

You should end up with a DataFrame that looks something like the following:
```
                     c_1	     c_2	       c_3	       c_4	      c_5	   ...
Alabama-Autauga	-0.060274	0.160527	-0.088356	 0.120480	-0.010824	...
Alabama-Baldwin	-0.149684	0.185969	-0.145743	-0.023092	-0.068677	...
Alabama-Barbour	0.506202	 0.296662	 0.146258	 0.297829	0.093111	...
...
```


```python
# create dimensionality-reduced data
def create_transformed_df(train_pca, counties_scaled, n_top_components):
    ''' Return a dataframe of data points with component features. 
        The dataframe should be indexed by State-County and contain component values.
        :param train_pca: A list of pca training data, returned by a PCA model.
        :param counties_scaled: A dataframe of normalized, original features.
        :param n_top_components: An integer, the number of top components to use.
        :return: A dataframe, indexed by State-County, with n_top_component values as columns.        
     '''
    # create new dataframe to add data to
    counties_transformed=pd.DataFrame()

    # for each of our new, transformed data points
    # append the component values to the dataframe
    for data in train_pca:
        # get component values for each data point
        components=data.label['projection'].float32_tensor.values
        counties_transformed=counties_transformed.append([list(components)])

    # index by county, just like counties_scaled
    counties_transformed.index=counties_scaled.index

    # keep only the top n components
    start_idx = N_COMPONENTS - n_top_components
    counties_transformed = counties_transformed.iloc[:,start_idx:]
    
    # reverse columns, component order     
    return counties_transformed.iloc[:, ::-1]
    
```

Now we can create a dataset where each county is described by the top n principle components that we analyzed earlier. Each of these components is a linear combination of the original feature space. We can interpret each of these components by analyzing the makeup of the component, shown previously.

### Define the `top_n` components to use in this transformed data

Your code should return data, indexed by 'State-County' and with as many columns as `top_n` components.

You can also choose to add descriptive column names for this data; names that correspond to the component number or feature-level makeup.


```python
## Specify top n
top_n = 7

# call your function and create a new dataframe
counties_transformed = create_transformed_df(train_pca, counties_scaled, n_top_components=top_n)

# add descriptive columns
PCA_list=['c_1', 'c_2', 'c_3', 'c_4', 'c_5', 'c_6', 'c_7']
counties_transformed.columns=PCA_list 


# print result
counties_transformed.head()
```




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
      <th>c_1</th>
      <th>c_2</th>
      <th>c_3</th>
      <th>c_4</th>
      <th>c_5</th>
      <th>c_6</th>
      <th>c_7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Alabama-Autauga</th>
      <td>-0.060274</td>
      <td>0.160527</td>
      <td>-0.088356</td>
      <td>0.120480</td>
      <td>-0.010824</td>
      <td>0.040452</td>
      <td>0.025895</td>
    </tr>
    <tr>
      <th>Alabama-Baldwin</th>
      <td>-0.149684</td>
      <td>0.185969</td>
      <td>-0.145743</td>
      <td>-0.023092</td>
      <td>-0.068677</td>
      <td>0.051573</td>
      <td>0.048137</td>
    </tr>
    <tr>
      <th>Alabama-Barbour</th>
      <td>0.506202</td>
      <td>0.296662</td>
      <td>0.146258</td>
      <td>0.297829</td>
      <td>0.093111</td>
      <td>-0.065244</td>
      <td>0.107730</td>
    </tr>
    <tr>
      <th>Alabama-Bibb</th>
      <td>0.069224</td>
      <td>0.190861</td>
      <td>0.224402</td>
      <td>0.011757</td>
      <td>0.283526</td>
      <td>0.017874</td>
      <td>-0.092053</td>
    </tr>
    <tr>
      <th>Alabama-Blount</th>
      <td>-0.091030</td>
      <td>0.254403</td>
      <td>0.022714</td>
      <td>-0.193824</td>
      <td>0.100738</td>
      <td>0.209945</td>
      <td>-0.005099</td>
    </tr>
  </tbody>
</table>
</div>



### Delete the Endpoint!

Now that we've deployed the mode and created our new, transformed training data, we no longer need the PCA endpoint.

As a clean up step, you should always delete your endpoints after you are done using them (and if you do not plan to deploy them to a website, for example).


```python
# delete predictor endpoint
session.delete_endpoint(pca_predictor.endpoint)
```

---
# Population Segmentation 

Now, you’ll use the unsupervised clustering algorithm, k-means, to segment counties using their PCA attributes, which are in the transformed DataFrame we just created. K-means is a clustering algorithm that identifies clusters of similar data points based on their component makeup. Since we have ~3000 counties and 34 attributes in the original dataset, the large feature space may have made it difficult to cluster the counties effectively. Instead, we have reduced the feature space to 7 PCA components, and we’ll cluster on this transformed dataset.

### EXERCISE: Define a k-means model

Your task will be to instantiate a k-means model. A `KMeans` estimator requires a number of parameters to be instantiated, which allow us to specify the type of training instance to use, and the model hyperparameters. 

You can read about the required parameters, in the [`KMeans` documentation](https://sagemaker.readthedocs.io/en/stable/kmeans.html); note that not all of the possible parameters are required.


### Choosing a "Good" K

One method for choosing a "good" k, is to choose based on empirical data. A bad k would be one so *high* that only one or two very close data points are near it, and another bad k would be one so *low* that data points are really far away from the centers.

You want to select a k such that data points in a single cluster are close together but that there are enough clusters to effectively separate the data. You can approximate this separation by measuring how close your data points are to each cluster center; the average centroid distance between cluster points and a centroid. After trying several values for k, the centroid distance typically reaches some "elbow"; it stops decreasing at a sharp rate and this indicates a good value of k. The graph below indicates the average centroid distance for value of k between 5 and 12.

<img src='notebook_ims/elbow_graph.png' width=50% />

A distance elbow can be seen around 8 when the distance starts to increase and then decrease at a slower rate. This indicates that there is enough separation to distinguish the data points in each cluster, but also that you included enough clusters so that the data points aren’t *extremely* far away from each cluster.


```python
# define a KMeans estimator
from sagemaker import KMeans

NUM_CLUSTERS = 8

kmeans = KMeans(role=role,
                train_instance_count=1,
                train_instance_type='ml.c4.xlarge',
                output_path=output_path, # using the same output path as was defined, earlier              
                k=NUM_CLUSTERS)

```

    train_instance_count has been renamed in sagemaker>=2.
    See: https://sagemaker.readthedocs.io/en/stable/v2.html for details.
    train_instance_type has been renamed in sagemaker>=2.
    See: https://sagemaker.readthedocs.io/en/stable/v2.html for details.


### EXERCISE: Create formatted, k-means training data

Just as before, you should convert the `counties_transformed` df into a numpy array and then into a RecordSet. This is the required format for passing training data into a `KMeans` model.


```python
# convert the transformed dataframe into record_set data
kmeans_train_data_np = counties_transformed.values.astype('float32')
kmeans_formatted_data = kmeans.record_set(kmeans_train_data_np)

```

### EXERCISE: Train the k-means model

Pass in the formatted training data and train the k-means model.


```python
%%time
# train kmeans
# train kmeans
kmeans.fit(kmeans_formatted_data)
```

    Defaulting to the only supported framework/algorithm version: 1. Ignoring framework/algorithm version: 1.
    Defaulting to the only supported framework/algorithm version: 1. Ignoring framework/algorithm version: 1.


    2021-05-16 12:23:57 Starting - Starting the training job...
    2021-05-16 12:24:21 Starting - Launching requested ML instancesProfilerReport-1621167837: InProgress
    .........
    2021-05-16 12:25:41 Starting - Preparing the instances for training......
    2021-05-16 12:26:41 Downloading - Downloading input data...
    2021-05-16 12:27:22 Training - Downloading the training image...
    2021-05-16 12:27:57 Uploading - Uploading generated training model
    2021-05-16 12:27:57 Completed - Training job completed
    [34mDocker entrypoint called with argument(s): train[0m
    [34mRunning default environment configuration script[0m
    [34m[05/16/2021 12:27:46 INFO 140141566281536] Reading default configuration from /opt/amazon/lib/python3.7/site-packages/algorithm/resources/default-input.json: {'init_method': 'random', 'mini_batch_size': '5000', 'epochs': '1', 'extra_center_factor': 'auto', 'local_lloyd_max_iter': '300', 'local_lloyd_tol': '0.0001', 'local_lloyd_init_method': 'kmeans++', 'local_lloyd_num_trials': 'auto', 'half_life_time_size': '0', 'eval_metrics': '["msd"]', 'force_dense': 'true', '_disable_wait_to_read': 'false', '_enable_profiler': 'false', '_kvstore': 'auto', '_log_level': 'info', '_num_gpus': 'auto', '_num_kv_servers': '1', '_num_slices': '1', '_tuning_objective_metric': ''}[0m
    [34m[05/16/2021 12:27:46 INFO 140141566281536] Merging with provided configuration from /opt/ml/input/config/hyperparameters.json: {'feature_dim': '7', 'k': '8', 'force_dense': 'True'}[0m
    [34m[05/16/2021 12:27:46 INFO 140141566281536] Final configuration: {'init_method': 'random', 'mini_batch_size': '5000', 'epochs': '1', 'extra_center_factor': 'auto', 'local_lloyd_max_iter': '300', 'local_lloyd_tol': '0.0001', 'local_lloyd_init_method': 'kmeans++', 'local_lloyd_num_trials': 'auto', 'half_life_time_size': '0', 'eval_metrics': '["msd"]', 'force_dense': 'True', '_disable_wait_to_read': 'false', '_enable_profiler': 'false', '_kvstore': 'auto', '_log_level': 'info', '_num_gpus': 'auto', '_num_kv_servers': '1', '_num_slices': '1', '_tuning_objective_metric': '', 'feature_dim': '7', 'k': '8'}[0m
    [34m[05/16/2021 12:27:46 WARNING 140141566281536] Loggers have already been setup.[0m
    [34mProcess 1 is a worker.[0m
    [34m[05/16/2021 12:27:46 INFO 140141566281536] Using default worker.[0m
    [34m[05/16/2021 12:27:46 INFO 140141566281536] Loaded iterator creator application/x-recordio-protobuf for content type ('application/x-recordio-protobuf', '1.0')[0m
    [34m[05/16/2021 12:27:46 INFO 140141566281536] Create Store: local[0m
    [34m[05/16/2021 12:27:47 INFO 140141566281536] nvidia-smi: took 0.030 seconds to run.[0m
    [34m[05/16/2021 12:27:47 INFO 140141566281536] nvidia-smi identified 0 GPUs.[0m
    [34m[05/16/2021 12:27:47 INFO 140141566281536] Number of GPUs being used: 0[0m
    [34m[05/16/2021 12:27:47 INFO 140141566281536] Checkpoint loading and saving are disabled.[0m
    [34m[05/16/2021 12:27:47 INFO 140141566281536] Setting up with params: {'init_method': 'random', 'mini_batch_size': '5000', 'epochs': '1', 'extra_center_factor': 'auto', 'local_lloyd_max_iter': '300', 'local_lloyd_tol': '0.0001', 'local_lloyd_init_method': 'kmeans++', 'local_lloyd_num_trials': 'auto', 'half_life_time_size': '0', 'eval_metrics': '["msd"]', 'force_dense': 'True', '_disable_wait_to_read': 'false', '_enable_profiler': 'false', '_kvstore': 'auto', '_log_level': 'info', '_num_gpus': 'auto', '_num_kv_servers': '1', '_num_slices': '1', '_tuning_objective_metric': '', 'feature_dim': '7', 'k': '8'}[0m
    [34m[05/16/2021 12:27:47 INFO 140141566281536] 'extra_center_factor' was set to 'auto', evaluated to 10.[0m
    [34m[05/16/2021 12:27:47 INFO 140141566281536] Number of GPUs being used: 0[0m
    [34m[05/16/2021 12:27:47 INFO 140141566281536] number of center slices 1[0m
    [34m[05/16/2021 12:27:47 WARNING 140141566281536] Batch size 5000 is bigger than the first batch data. Effective batch size used to initialize is 3218[0m
    [34m#metrics {"StartTime": 1621168067.054921, "EndTime": 1621168067.0549648, "Dimensions": {"Algorithm": "AWS/KMeansWebscale", "Host": "algo-1", "Operation": "training", "Meta": "init_train_data_iter"}, "Metrics": {"Total Records Seen": {"sum": 3218.0, "count": 1, "min": 3218, "max": 3218}, "Total Batches Seen": {"sum": 1.0, "count": 1, "min": 1, "max": 1}, "Max Records Seen Between Resets": {"sum": 3218.0, "count": 1, "min": 3218, "max": 3218}, "Max Batches Seen Between Resets": {"sum": 1.0, "count": 1, "min": 1, "max": 1}, "Reset Count": {"sum": 0.0, "count": 1, "min": 0, "max": 0}, "Number of Records Since Last Reset": {"sum": 3218.0, "count": 1, "min": 3218, "max": 3218}, "Number of Batches Since Last Reset": {"sum": 1.0, "count": 1, "min": 1, "max": 1}}}
    [0m
    [34m[2021-05-16 12:27:47.055] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/train", "epoch": 0, "duration": 37, "num_examples": 1, "num_bytes": 167336}[0m
    [34m[2021-05-16 12:27:47.113] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/train", "epoch": 1, "duration": 56, "num_examples": 1, "num_bytes": 167336}[0m
    [34m[05/16/2021 12:27:47 INFO 140141566281536] processed a total of 3218 examples[0m
    [34m[05/16/2021 12:27:47 INFO 140141566281536] #progress_metric: host=algo-1, completed 100.0 % of epochs[0m
    [34m#metrics {"StartTime": 1621168067.0554194, "EndTime": 1621168067.1145475, "Dimensions": {"Algorithm": "AWS/KMeansWebscale", "Host": "algo-1", "Operation": "training", "epoch": 0, "Meta": "training_data_iter"}, "Metrics": {"Total Records Seen": {"sum": 6436.0, "count": 1, "min": 6436, "max": 6436}, "Total Batches Seen": {"sum": 2.0, "count": 1, "min": 2, "max": 2}, "Max Records Seen Between Resets": {"sum": 3218.0, "count": 1, "min": 3218, "max": 3218}, "Max Batches Seen Between Resets": {"sum": 1.0, "count": 1, "min": 1, "max": 1}, "Reset Count": {"sum": 1.0, "count": 1, "min": 1, "max": 1}, "Number of Records Since Last Reset": {"sum": 3218.0, "count": 1, "min": 3218, "max": 3218}, "Number of Batches Since Last Reset": {"sum": 1.0, "count": 1, "min": 1, "max": 1}}}
    [0m
    [34m[05/16/2021 12:27:47 INFO 140141566281536] #throughput_metric: host=algo-1, train throughput=54276.96611628075 records/second[0m
    [34m[05/16/2021 12:27:47 WARNING 140141566281536] wait_for_all_workers will not sync workers since the kv store is not running distributed[0m
    [34m[05/16/2021 12:27:47 INFO 140141566281536] shrinking 80 centers into 8[0m
    [34m[05/16/2021 12:27:47 INFO 140141566281536] local kmeans attempt #0. Current mean square distance 0.069973[0m
    [34m[05/16/2021 12:27:47 INFO 140141566281536] local kmeans attempt #1. Current mean square distance 0.066166[0m
    [34m[05/16/2021 12:27:47 INFO 140141566281536] local kmeans attempt #2. Current mean square distance 0.065614[0m
    [34m[05/16/2021 12:27:47 INFO 140141566281536] local kmeans attempt #3. Current mean square distance 0.071123[0m
    [34m[05/16/2021 12:27:47 INFO 140141566281536] local kmeans attempt #4. Current mean square distance 0.066850[0m
    [34m[05/16/2021 12:27:47 INFO 140141566281536] local kmeans attempt #5. Current mean square distance 0.066086[0m
    [34m[05/16/2021 12:27:47 INFO 140141566281536] local kmeans attempt #6. Current mean square distance 0.068548[0m
    [34m[05/16/2021 12:27:47 INFO 140141566281536] local kmeans attempt #7. Current mean square distance 0.064049[0m
    [34m[05/16/2021 12:27:47 INFO 140141566281536] local kmeans attempt #8. Current mean square distance 0.076572[0m
    [34m[05/16/2021 12:27:47 INFO 140141566281536] local kmeans attempt #9. Current mean square distance 0.068465[0m
    [34m[05/16/2021 12:27:47 INFO 140141566281536] finished shrinking process. Mean Square Distance = 0[0m
    [34m[05/16/2021 12:27:47 INFO 140141566281536] #quality_metric: host=algo-1, train msd <loss>=0.06404909491539001[0m
    [34m[05/16/2021 12:27:47 INFO 140141566281536] compute all data-center distances: inner product took: 40.6378%, (0.026449 secs)[0m
    [34m[05/16/2021 12:27:47 INFO 140141566281536] collect from kv store took: 10.7463%, (0.006994 secs)[0m
    [34m[05/16/2021 12:27:47 INFO 140141566281536] splitting centers key-value pair took: 10.5613%, (0.006874 secs)[0m
    [34m[05/16/2021 12:27:47 INFO 140141566281536] compute all data-center distances: point norm took: 8.8301%, (0.005747 secs)[0m
    [34m[05/16/2021 12:27:47 INFO 140141566281536] batch data loading with context took: 8.2019%, (0.005338 secs)[0m
    [34m[05/16/2021 12:27:47 INFO 140141566281536] predict compute msd took: 6.9241%, (0.004507 secs)[0m
    [34m[05/16/2021 12:27:47 INFO 140141566281536] gradient: one_hot took: 6.7758%, (0.004410 secs)[0m
    [34m[05/16/2021 12:27:47 INFO 140141566281536] update state and report convergance took: 2.9137%, (0.001896 secs)[0m
    [34m[05/16/2021 12:27:47 INFO 140141566281536] gradient: cluster size  took: 1.9935%, (0.001297 secs)[0m
    [34m[05/16/2021 12:27:47 INFO 140141566281536] gradient: cluster center took: 1.8488%, (0.001203 secs)[0m
    [34m[05/16/2021 12:27:47 INFO 140141566281536] update set-up time took: 0.3341%, (0.000217 secs)[0m
    [34m[05/16/2021 12:27:47 INFO 140141566281536] compute all data-center distances: center norm took: 0.2029%, (0.000132 secs)[0m
    [34m[05/16/2021 12:27:47 INFO 140141566281536] predict minus dist took: 0.0297%, (0.000019 secs)[0m
    [34m[05/16/2021 12:27:47 INFO 140141566281536] TOTAL took: 0.06508517265319824[0m
    [34m[05/16/2021 12:27:47 INFO 140141566281536] Number of GPUs being used: 0[0m
    [34m#metrics {"StartTime": 1621168067.0175502, "EndTime": 1621168067.4292173, "Dimensions": {"Algorithm": "AWS/KMeansWebscale", "Host": "algo-1", "Operation": "training"}, "Metrics": {"initialize.time": {"sum": 29.725313186645508, "count": 1, "min": 29.725313186645508, "max": 29.725313186645508}, "epochs": {"sum": 1.0, "count": 1, "min": 1, "max": 1}, "update.time": {"sum": 58.919429779052734, "count": 1, "min": 58.919429779052734, "max": 58.919429779052734}, "_shrink.time": {"sum": 310.76788902282715, "count": 1, "min": 310.76788902282715, "max": 310.76788902282715}, "finalize.time": {"sum": 312.53957748413086, "count": 1, "min": 312.53957748413086, "max": 312.53957748413086}, "model.serialize.time": {"sum": 1.7006397247314453, "count": 1, "min": 1.7006397247314453, "max": 1.7006397247314453}}}
    [0m
    [34m[05/16/2021 12:27:47 INFO 140141566281536] Test data is not provided.[0m
    [34m#metrics {"StartTime": 1621168067.4293017, "EndTime": 1621168067.4295528, "Dimensions": {"Algorithm": "AWS/KMeansWebscale", "Host": "algo-1", "Operation": "training"}, "Metrics": {"setuptime": {"sum": 11.597871780395508, "count": 1, "min": 11.597871780395508, "max": 11.597871780395508}, "totaltime": {"sum": 476.8693447113037, "count": 1, "min": 476.8693447113037, "max": 476.8693447113037}}}
    [0m
    Training seconds: 85
    Billable seconds: 85
    CPU times: user 548 ms, sys: 26.7 ms, total: 574 ms
    Wall time: 4min 12s


### EXERCISE: Deploy the k-means model

Deploy the trained model to create a `kmeans_predictor`.



```python
%%time
# deploy the model to create a predictor
kmeans_predictor = kmeans.deploy(initial_instance_count=1, 
                                 instance_type='ml.t2.medium')
```

    Defaulting to the only supported framework/algorithm version: 1. Ignoring framework/algorithm version: 1.


    -------------------!CPU times: user 356 ms, sys: 5.59 ms, total: 361 ms
    Wall time: 9min 32s


### EXERCISE: Pass in the training data and assign predicted cluster labels

After deploying the model, you can pass in the k-means training data, as a numpy array, and get resultant, predicted cluster labels for each data point.


```python
# get the predicted clusters for all the kmeans training data
cluster_info=kmeans_predictor.predict(kmeans_train_data_np)
```

## Exploring the resultant clusters

The resulting predictions should give you information about the cluster that each data point belongs to.

You should be able to answer the **question**: which cluster does a given data point belong to?


```python
# print cluster info for first data point
data_idx = 0

print('County is: ', counties_transformed.index[data_idx])
print()
print(cluster_info[data_idx])
```

    County is:  Alabama-Autauga
    
    label {
      key: "closest_cluster"
      value {
        float32_tensor {
          values: 2.0
        }
      }
    }
    label {
      key: "distance_to_cluster"
      value {
        float32_tensor {
          values: 0.23653684556484222
        }
      }
    }
    


### Visualize the distribution of data over clusters

Get the cluster labels for each of our data points (counties) and visualize the distribution of points over each cluster.


```python
# get all cluster labels
cluster_labels = [c.label['closest_cluster'].float32_tensor.values[0] for c in cluster_info]
```


```python
# count up the points in each cluster
cluster_df = pd.DataFrame(cluster_labels)[0].value_counts()

print(cluster_df)
```

    0.0    868
    6.0    824
    2.0    481
    3.0    357
    4.0    303
    1.0    258
    5.0     96
    7.0     31
    Name: 0, dtype: int64


Now, you may be wondering, what do each of these clusters tell us about these data points? To improve explainability, we need to access the underlying model to get the cluster centers. These centers will help describe which features characterize each cluster.

### Delete the Endpoint!

Now that you've deployed the k-means model and extracted the cluster labels for each data point, you no longer need the k-means endpoint.


```python
# delete kmeans endpoint
session.delete_endpoint(kmeans_predictor.endpoint)
```

---
# Model Attributes & Explainability

Explaining the result of the modeling is an important step in making use of our analysis. By combining PCA and k-means, and the information contained in the model attributes within a SageMaker trained model, you can learn about a population and remark on some patterns you've found, based on the data.

### EXERCISE: Access the k-means model attributes

Extract the k-means model attributes from where they are saved as a TAR file in an S3 bucket.

You'll need to access the model by the k-means training job name, and then unzip the file into `model_algo-1`. Then you can load that file using MXNet, as before.


```python
# download and unzip the kmeans model file
kmeans_job_name = 'kmeans-2021-05-16-12-23-57-384'

model_key = os.path.join(prefix, kmeans_job_name, 'output/model.tar.gz')

# download the model file
boto3.resource('s3').Bucket(bucket_name).download_file(model_key, 'model.tar.gz')
os.system('tar -zxvf model.tar.gz')
os.system('unzip model_algo-1')
```




    2304




```python
# get the trained kmeans params using mxnet
kmeans_model_params = mx.ndarray.load('model_algo-1')

print(kmeans_model_params)
```

    [
    [[-0.04160065  0.10482064  0.14324667 -0.05932264 -0.02128213  0.05524607
      -0.02234111]
     [ 0.30582225 -0.22351    -0.08026904 -0.1384553   0.11934531 -0.10986733
      -0.08797634]
     [-0.12036126  0.08424732 -0.29536614  0.0692042  -0.03315074  0.04189404
       0.00270831]
     [ 0.41052106  0.25037077  0.10011049  0.28726736  0.1084631  -0.05808229
       0.04905725]
     [-0.24429514 -0.42835456  0.10017882  0.11415306  0.04280135 -0.01119636
       0.10603061]
     [ 1.236671   -0.26430753 -0.17885119 -0.391919   -0.10169901  0.09295962
       0.12522331]
     [-0.21635447 -0.01186692  0.00965554 -0.05123839 -0.02331246 -0.04484653
      -0.00996859]
     [ 0.7181207  -0.66568977  0.1996651   0.5270777  -0.38118798  0.08893491
      -0.3744459 ]]
    <NDArray 8x7 @cpu(0)>]


There is only 1 set of model parameters contained within the k-means model: the cluster centroid locations in PCA-transformed, component space.

* **centroids**: The location of the centers of each cluster in component space, identified by the k-means algorithm. 



```python
# get all the centroids
cluster_centroids=pd.DataFrame(kmeans_model_params[0].asnumpy())
cluster_centroids.columns=counties_transformed.columns

display(cluster_centroids)
```


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
      <th>c_1</th>
      <th>c_2</th>
      <th>c_3</th>
      <th>c_4</th>
      <th>c_5</th>
      <th>c_6</th>
      <th>c_7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.041601</td>
      <td>0.104821</td>
      <td>0.143247</td>
      <td>-0.059323</td>
      <td>-0.021282</td>
      <td>0.055246</td>
      <td>-0.022341</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.305822</td>
      <td>-0.223510</td>
      <td>-0.080269</td>
      <td>-0.138455</td>
      <td>0.119345</td>
      <td>-0.109867</td>
      <td>-0.087976</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.120361</td>
      <td>0.084247</td>
      <td>-0.295366</td>
      <td>0.069204</td>
      <td>-0.033151</td>
      <td>0.041894</td>
      <td>0.002708</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.410521</td>
      <td>0.250371</td>
      <td>0.100110</td>
      <td>0.287267</td>
      <td>0.108463</td>
      <td>-0.058082</td>
      <td>0.049057</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.244295</td>
      <td>-0.428355</td>
      <td>0.100179</td>
      <td>0.114153</td>
      <td>0.042801</td>
      <td>-0.011196</td>
      <td>0.106031</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.236671</td>
      <td>-0.264308</td>
      <td>-0.178851</td>
      <td>-0.391919</td>
      <td>-0.101699</td>
      <td>0.092960</td>
      <td>0.125223</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.216354</td>
      <td>-0.011867</td>
      <td>0.009656</td>
      <td>-0.051238</td>
      <td>-0.023312</td>
      <td>-0.044847</td>
      <td>-0.009969</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.718121</td>
      <td>-0.665690</td>
      <td>0.199665</td>
      <td>0.527078</td>
      <td>-0.381188</td>
      <td>0.088935</td>
      <td>-0.374446</td>
    </tr>
  </tbody>
</table>
</div>


### Visualizing Centroids in Component Space

You can't visualize 7-dimensional centroids in space, but you can plot a heatmap of the centroids and their location in the transformed feature space. 

This gives you insight into what characteristics define each cluster. Often with unsupervised learning, results are hard to interpret. This is one way to make use of the results of PCA + clustering techniques, together. Since you were able to examine the makeup of each PCA component, you can understand what each centroid represents in terms of the PCA components.


```python
# generate a heatmap in component space, using the seaborn library
plt.figure(figsize = (12,9))
ax = sns.heatmap(cluster_centroids.T, cmap = 'YlGnBu')
ax.set_xlabel("Cluster")
plt.yticks(fontsize = 16)
plt.xticks(fontsize = 16)
ax.set_title("Attribute Value by Centroid")
plt.show()
```


    
![png](output_99_0.png)
    


If you've forgotten what each component corresponds to at an original-feature-level, that's okay! You can use the previously defined `display_component` function to see the feature-level makeup.


```python
# what do each of these components mean again?
# let's use the display function, from above
component_num=7
display_component(v, counties_scaled.columns.values, component_num=component_num)
```


    
![png](output_101_0.png)
    


### Natural Groupings

You can also map the cluster labels back to each individual county and examine which counties are naturally grouped together.


```python
# add a 'labels' column to the dataframe
counties_transformed['labels']=list(map(int, cluster_labels))

# sort by cluster label 0-6
sorted_counties = counties_transformed.sort_values('labels', ascending=True)
# view some pts in cluster 0
sorted_counties.head(20)
```




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
      <th>c_1</th>
      <th>c_2</th>
      <th>c_3</th>
      <th>c_4</th>
      <th>c_5</th>
      <th>c_6</th>
      <th>c_7</th>
      <th>labels</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Michigan-Montcalm</th>
      <td>-0.093800</td>
      <td>0.277402</td>
      <td>0.118879</td>
      <td>-0.183504</td>
      <td>0.011605</td>
      <td>0.125444</td>
      <td>-0.056001</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Missouri-Ste. Genevieve</th>
      <td>-0.237802</td>
      <td>0.167400</td>
      <td>0.156479</td>
      <td>-0.199382</td>
      <td>0.166887</td>
      <td>-0.027151</td>
      <td>-0.148141</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Missouri-St. Clair</th>
      <td>-0.027264</td>
      <td>0.066711</td>
      <td>0.254064</td>
      <td>-0.086178</td>
      <td>-0.152424</td>
      <td>0.123932</td>
      <td>0.064403</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Tennessee-Smith</th>
      <td>-0.196535</td>
      <td>0.148426</td>
      <td>0.138396</td>
      <td>-0.138055</td>
      <td>0.055282</td>
      <td>0.056815</td>
      <td>0.106468</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Indiana-Cass</th>
      <td>-0.049903</td>
      <td>0.162810</td>
      <td>0.080575</td>
      <td>-0.184691</td>
      <td>-0.017748</td>
      <td>-0.158167</td>
      <td>-0.103680</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Missouri-Ripley</th>
      <td>-0.018762</td>
      <td>0.071493</td>
      <td>0.318636</td>
      <td>-0.092893</td>
      <td>-0.091563</td>
      <td>0.088709</td>
      <td>-0.103954</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Indiana-Clay</th>
      <td>-0.218904</td>
      <td>0.160771</td>
      <td>0.099574</td>
      <td>-0.120858</td>
      <td>-0.036950</td>
      <td>0.031927</td>
      <td>-0.098949</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Indiana-Clinton</th>
      <td>-0.138011</td>
      <td>0.248154</td>
      <td>0.054567</td>
      <td>-0.286209</td>
      <td>0.161532</td>
      <td>-0.179524</td>
      <td>-0.121162</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Indiana-Crawford</th>
      <td>-0.120441</td>
      <td>0.244305</td>
      <td>0.219173</td>
      <td>-0.168967</td>
      <td>-0.025639</td>
      <td>0.135150</td>
      <td>0.028699</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Missouri-Reynolds</th>
      <td>-0.031156</td>
      <td>0.021333</td>
      <td>0.381899</td>
      <td>-0.074204</td>
      <td>0.042624</td>
      <td>0.045821</td>
      <td>-0.260713</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Missouri-Ray</th>
      <td>-0.179584</td>
      <td>0.234674</td>
      <td>0.063161</td>
      <td>-0.144645</td>
      <td>0.040486</td>
      <td>0.140745</td>
      <td>-0.055201</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Missouri-Randolph</th>
      <td>-0.094683</td>
      <td>0.173096</td>
      <td>0.103851</td>
      <td>-0.093550</td>
      <td>-0.130811</td>
      <td>-0.048471</td>
      <td>0.060319</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Missouri-Polk</th>
      <td>-0.139436</td>
      <td>0.016433</td>
      <td>0.143301</td>
      <td>-0.066795</td>
      <td>-0.093818</td>
      <td>0.100575</td>
      <td>0.016443</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Indiana-Blackford</th>
      <td>-0.207763</td>
      <td>0.335245</td>
      <td>0.165504</td>
      <td>-0.221740</td>
      <td>-0.042952</td>
      <td>-0.116686</td>
      <td>-0.017118</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Indiana-Delaware</th>
      <td>-0.033202</td>
      <td>0.160756</td>
      <td>0.063104</td>
      <td>-0.017805</td>
      <td>-0.264668</td>
      <td>-0.034917</td>
      <td>-0.049219</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Indiana-Elkhart</th>
      <td>-0.046948</td>
      <td>0.255348</td>
      <td>-0.012383</td>
      <td>-0.211758</td>
      <td>0.033895</td>
      <td>-0.254342</td>
      <td>-0.097961</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Indiana-Fayette</th>
      <td>-0.097968</td>
      <td>0.320962</td>
      <td>0.156540</td>
      <td>-0.175092</td>
      <td>-0.111585</td>
      <td>0.025470</td>
      <td>-0.037197</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Missouri-Pike</th>
      <td>-0.154928</td>
      <td>0.072187</td>
      <td>0.110249</td>
      <td>-0.070933</td>
      <td>0.045705</td>
      <td>-0.015308</td>
      <td>0.017677</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Tennessee-Anderson</th>
      <td>-0.109347</td>
      <td>0.111530</td>
      <td>0.014941</td>
      <td>-0.021213</td>
      <td>-0.195134</td>
      <td>0.034839</td>
      <td>0.079956</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Virginia-Lee</th>
      <td>0.066950</td>
      <td>0.131093</td>
      <td>0.231805</td>
      <td>-0.027617</td>
      <td>-0.199608</td>
      <td>0.222746</td>
      <td>0.028104</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



You can also examine one of the clusters in more detail, like cluster 1, for example. A quick glance at the location of the centroid in component space (the heatmap) tells us that it has the highest value for the `comp_6` attribute. You can now see which counties fit that description.


```python
# get all counties with label == 1
cluster=counties_transformed[counties_transformed['labels']==1]
cluster.head()
```




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
      <th>c_1</th>
      <th>c_2</th>
      <th>c_3</th>
      <th>c_4</th>
      <th>c_5</th>
      <th>c_6</th>
      <th>c_7</th>
      <th>labels</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Alaska-Aleutians East Borough</th>
      <td>0.214891</td>
      <td>-0.326237</td>
      <td>-0.087370</td>
      <td>0.183295</td>
      <td>0.309654</td>
      <td>-0.882860</td>
      <td>-0.502769</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Alaska-Aleutians West Census Area</th>
      <td>-0.010280</td>
      <td>-0.263201</td>
      <td>-0.366322</td>
      <td>0.147377</td>
      <td>0.268785</td>
      <td>-0.687394</td>
      <td>-0.453934</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Alaska-Kodiak Island Borough</th>
      <td>-0.008272</td>
      <td>-0.367003</td>
      <td>-0.162269</td>
      <td>0.191822</td>
      <td>0.042486</td>
      <td>-0.298352</td>
      <td>-0.429001</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Alaska-North Slope Borough</th>
      <td>0.004283</td>
      <td>-0.637046</td>
      <td>-0.218852</td>
      <td>0.349359</td>
      <td>0.292917</td>
      <td>-0.406611</td>
      <td>-0.702960</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Arizona-Cochise</th>
      <td>0.276746</td>
      <td>-0.261673</td>
      <td>-0.142366</td>
      <td>0.022962</td>
      <td>-0.198940</td>
      <td>0.009623</td>
      <td>-0.091299</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Final Cleanup!

* Double check that you have deleted all your endpoints.
* I'd also suggest manually deleting your S3 bucket, models, and endpoint configurations directly from your AWS console.

You can find thorough cleanup instructions, [in the documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/ex1-cleanup.html).

---
# Conclusion

You have just walked through a machine learning workflow for unsupervised learning, specifically, for clustering a dataset using k-means after reducing the dimensionality using PCA. By accessing the underlying models created within  SageMaker, you were able to improve the explainability of your model and draw insights from the resultant clusters. 

Using these techniques, you have been able to better understand the essential characteristics of different counties in the US and segment them into similar groups, accordingly.
