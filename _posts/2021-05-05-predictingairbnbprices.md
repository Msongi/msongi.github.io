---
title: "Predicting AirBnB Prices and Review scores"
date: 2021-05-05
tags: [data wrangling, data science, machine learning]
header:
  image: "/images/perceptron/percept.jpg"
excerpt: "Data Wrangling, Data Science, Machine learning"
mathjax: "true"
---
# Predicting AirBnB Prices and Review scores

## Business Understanding

Since 2008, guests and hosts have used Airbnb to travel in a more unique, personalized way. As part of its growth initiative, AirBnB is seeking to enhance customer experience and improve business. These insights will help Customer Experience, Business Development and Marketing departments. We will monitor local trends and gain strategic insight on the growth of AirBnB in seattle and on ways of improving the customer experience.

#### Data mining objectives

AirBnb is seeking to know what vibes are happening in each neighbourhood, if there is a general upward trend of new AirBnb listings in Seattle, which factors influence price and customer experience. We want to find if there are metrics which the business can associate with good prices and increased listings.Also to find a good way to measure customer satisfaction and ways that AirBnb hosts can do to get more bookings.

We will use the listings, calendar and reviews datasets for our analysis. We will wrangle and explore this data to understand what data we have and find insights from it to meet our objectives. A large part of this will involve applying transformations to the data, and dealing with missing data.


To summarise, translating our business goals into data mining goals, we have:       
    
    1. What features of the listings are related to price?    
    2. What vibes are happening in each neighbourhood?    
    3. Are there factors/metrics for that help understand customer experience?    
    4. Are there things that guests/AirBnB can do to boost guest experience and get more clients?
    5. Has there been an increase in the number of AirBnb hosts?
    6. What are the busiest times of the year in seattle and by how much do the prices change

clarify who customer and host is
    



#### Data mining problem type

We will mine the data to answer or attempt to answer the data mining objectives. The question of vibes happening in each neighhbourhood,factors that help understand/improve customer experience will require exploratory data analysis. The EDA will be perfomed on features on the listings and review data sets to find the metrics.

The question of factors related to price and guest experience will become either a regression or classification problem for which we will do the modelling. The challenges will be selecting the features that are important and preparing the data for modelling (removing outliers, imputing missing values etc).

To answer the question on price and customer experience will deal with feature engineering. Choosing which features will be used and according to the chosen model, determine the parameters.


#### Criteria for model assessment

The model will be evaluated against tests to ensure its valid. 

The first data mining problems will be approached using descriptive statistics and visualisations of the distributions. By these we will be able to ascertain key metrics and some feature importances.

We will also need to ensure that the metric chosen will satisfy the business requirements for being an adequate representation of customer experience, and that it is not just chosen solely to produce good results from the model (Feature importance)

For the second data mining problem, we will use cross validation to evaluate candidates for modelling the data. We will evaluate using the R2 score and the RMSE if using regression. We could also plot residuals of the model to evaluate skew.

The third data mining problem will follow from the solution of the second data mining problem. We can apply a properly evaluated model on the data in order to assess the feature importance. We will ensure that the features chosen from the feature importance are appropriate for using in a strategy that could conceivably target hosts and how they could improve the guests' experiences.

Finally, it is important to note the broad nature of the project. We will narrow it down to improve customer service. We are only looking to improve guests' customer experience, and this could be limited to certain subsets of the user base. As long as the user base chosen is large enough for potential insights to be significant, then this will contribute towards satisfying our business goals.

#### Action plan

Boiling down the above into a plan, we have:

   1.Explore and understand the data.

   2.Perform exploratory and descriptive analysis in order to find an appropriate response
   variable related to customer experience and price

   3.Based on whether the response variable is continous or discrete either frame a regression
   or classification problem for predicting this response variable.

   4.Clean and tidy the data set in order to prepare it for modelling.

   5.Select candidate algorithms for modelling the data and evaluate them, utilizing cross-validation.

   6.Use the prediction model to ascertain the most important aspects that hosts can improve guests' experience.

   7.Communicate the results as key points that a non-technical business audience could understand.



## Data Understanding


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re

```


```python
!pip install wordcloud
```

    Requirement already satisfied: wordcloud in /home/fucking/anaconda/lib/python3.8/site-packages (1.8.1)
    Requirement already satisfied: pillow in /home/fucking/anaconda/lib/python3.8/site-packages (from wordcloud) (8.0.1)
    Requirement already satisfied: numpy>=1.6.1 in /home/fucking/anaconda/lib/python3.8/site-packages (from wordcloud) (1.19.2)
    Requirement already satisfied: matplotlib in /home/fucking/anaconda/lib/python3.8/site-packages (from wordcloud) (3.3.2)
    Requirement already satisfied: certifi>=2020.06.20 in /home/fucking/anaconda/lib/python3.8/site-packages (from matplotlib->wordcloud) (2020.6.20)
    Requirement already satisfied: kiwisolver>=1.0.1 in /home/fucking/anaconda/lib/python3.8/site-packages (from matplotlib->wordcloud) (1.3.0)
    Requirement already satisfied: python-dateutil>=2.1 in /home/fucking/anaconda/lib/python3.8/site-packages (from matplotlib->wordcloud) (2.8.1)
    Requirement already satisfied: cycler>=0.10 in /home/fucking/anaconda/lib/python3.8/site-packages (from matplotlib->wordcloud) (0.10.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.3 in /home/fucking/anaconda/lib/python3.8/site-packages (from matplotlib->wordcloud) (2.4.7)
    Requirement already satisfied: six>=1.5 in /home/fucking/anaconda/lib/python3.8/site-packages (from python-dateutil>=2.1->matplotlib->wordcloud) (1.15.0)



```python
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     /home/fucking/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package punkt to /home/fucking/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package wordnet to /home/fucking/nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!





    True



### Data Gathering


```python
calender_df=pd.read_csv("calendar.csv", delimiter=',')
listings_df=pd.read_csv("listings.csv", delimiter=',')
reviews_df=pd.read_csv("reviews.csv", delimiter=',')
```

### Understanding the calender dataset


```python
calender_df.shape
```




    (1393570, 4)




```python
calender_df.head()
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
      <th>listing_id</th>
      <th>date</th>
      <th>available</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>241032</td>
      <td>2016-01-04</td>
      <td>t</td>
      <td>$85.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>241032</td>
      <td>2016-01-05</td>
      <td>t</td>
      <td>$85.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>241032</td>
      <td>2016-01-06</td>
      <td>f</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>241032</td>
      <td>2016-01-07</td>
      <td>f</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>241032</td>
      <td>2016-01-08</td>
      <td>f</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Are there missing values
no_missing_vals_cols=set(calender_df.columns[calender_df.isnull().mean()==0])
no_missing_vals_cols
```




    {'available', 'date', 'listing_id'}




```python
#replace True or false with 1 and 0 for available not available and change 
calender_df=calender_df.replace(to_replace="t",value="1")
calender_df=calender_df.replace(to_replace="f",value="0")
calender_df['available'] = calender_df['available'].astype(int, errors = 'raise')
```


```python
#Find the proportion of available to unavailable  values
calender_df.available.value_counts()/ calender_df.shape[0]
```




    1    0.67061
    0    0.32939
    Name: available, dtype: float64




```python
calender_df['price'] = calender_df['price'].str.extract(r'(\d+)', expand=False)
# make price float
calender_df['price'] = calender_df['price'].astype(float, errors = 'raise')
```


```python
#fill missing price values with mean
calender_df['price'].fillna((calender_df['price'].mean()), inplace=True)
```


```python
#plot the proportions of available to not available
price_val=calender_df.groupby(['available']).listing_id.count()
price_val.plot.pie(figsize=(6,6))
```




    <AxesSubplot:ylabel='listing_id'>




    
![png](output_22_1.png)
    


67% of listings are available and 32.9% are unavailable 365 days.




```python
#check the calender data types
calender_df.dtypes
```




    listing_id      int64
    date           object
    available       int64
    price         float64
    dtype: object



The date column is saved as an object, we have to convert it to date time dtype for analysis


```python
#convert the date column from object to datetime
calender_df['date'] =  pd.to_datetime(calender_df['date'])
```


```python
calender_df.date.dtypes
```




    dtype('<M8[ns]')



A more interesting approach we will need is to split the date time into day of the week, week of the year, month and year. This will help us uncover insights into trends as relates to time


```python
# split date time into day of the week, week of the year, month of the year and year for indepth analysis

# Create new columns
calender_df['dayofweek'] = calender_df['date'].dt.dayofweek
calender_df['weekofyear'] = calender_df['date'].dt.isocalendar().week
calender_df['month'] = calender_df['date'].dt.month
calender_df['year'] = calender_df['date'].dt.year

calender_df[:5]
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
      <th>listing_id</th>
      <th>date</th>
      <th>available</th>
      <th>price</th>
      <th>dayofweek</th>
      <th>weekofyear</th>
      <th>month</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>241032</td>
      <td>2016-01-04</td>
      <td>1</td>
      <td>85.000000</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>1</th>
      <td>241032</td>
      <td>2016-01-05</td>
      <td>1</td>
      <td>85.000000</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>2</th>
      <td>241032</td>
      <td>2016-01-06</td>
      <td>0</td>
      <td>137.090652</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>3</th>
      <td>241032</td>
      <td>2016-01-07</td>
      <td>0</td>
      <td>137.090652</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>4</th>
      <td>241032</td>
      <td>2016-01-08</td>
      <td>0</td>
      <td>137.090652</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>2016</td>
    </tr>
  </tbody>
</table>
</div>



What we have gathered from the calender data is that listings are grouped into whether or not they are available, their price and date of whether they were available and not available. We learnt that there are 3818 listings in total, 67% available.

To note: We split the datetime into day of the week, week of the year, month and year. This will be useful for our EDA

### Understanding the reviews data

The reviews dataset is small and contains comments made by guests. By knowing frequently used words or words associated with quality, we provide insights for AirBnb hosts


```python
df2=reviews_df
df2.head()
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
      <th>listing_id</th>
      <th>id</th>
      <th>date</th>
      <th>reviewer_id</th>
      <th>reviewer_name</th>
      <th>comments</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7202016</td>
      <td>38917982</td>
      <td>2015-07-19</td>
      <td>28943674</td>
      <td>Bianca</td>
      <td>Cute and cozy place. Perfect location to every...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7202016</td>
      <td>39087409</td>
      <td>2015-07-20</td>
      <td>32440555</td>
      <td>Frank</td>
      <td>Kelly has a great room in a very central locat...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7202016</td>
      <td>39820030</td>
      <td>2015-07-26</td>
      <td>37722850</td>
      <td>Ian</td>
      <td>Very spacious apartment, and in a great neighb...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7202016</td>
      <td>40813543</td>
      <td>2015-08-02</td>
      <td>33671805</td>
      <td>George</td>
      <td>Close to Seattle Center and all it has to offe...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7202016</td>
      <td>41986501</td>
      <td>2015-08-10</td>
      <td>34959538</td>
      <td>Ming</td>
      <td>Kelly was a great host and very accommodating ...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2.shape
```




    (84849, 6)




```python
df2.dtypes
```




    listing_id        int64
    id                int64
    date             object
    reviewer_id       int64
    reviewer_name    object
    comments         object
    dtype: object




```python
#convert date to datetime
df2['date'] =  pd.to_datetime(df2['date'])
```


```python
# split date time into day of the week, week of the year, month of the year and year for indepth analysis
#calender_df.reset_index(inplace=True)

# Create new columns
df2['dayofweek'] = df2['date'].dt.dayofweek
df2['weekofyear'] = df2['date'].dt.isocalendar().week
df2['month'] = df2['date'].dt.month
df2['year'] = df2['date'].dt.year
#calender_df.drop('level_0',inplace=True, axis=1)
df2[:5]
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
      <th>listing_id</th>
      <th>id</th>
      <th>date</th>
      <th>reviewer_id</th>
      <th>reviewer_name</th>
      <th>comments</th>
      <th>dayofweek</th>
      <th>weekofyear</th>
      <th>month</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7202016</td>
      <td>38917982</td>
      <td>2015-07-19</td>
      <td>28943674</td>
      <td>Bianca</td>
      <td>Cute and cozy place. Perfect location to every...</td>
      <td>6</td>
      <td>29</td>
      <td>7</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7202016</td>
      <td>39087409</td>
      <td>2015-07-20</td>
      <td>32440555</td>
      <td>Frank</td>
      <td>Kelly has a great room in a very central locat...</td>
      <td>0</td>
      <td>30</td>
      <td>7</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7202016</td>
      <td>39820030</td>
      <td>2015-07-26</td>
      <td>37722850</td>
      <td>Ian</td>
      <td>Very spacious apartment, and in a great neighb...</td>
      <td>6</td>
      <td>30</td>
      <td>7</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7202016</td>
      <td>40813543</td>
      <td>2015-08-02</td>
      <td>33671805</td>
      <td>George</td>
      <td>Close to Seattle Center and all it has to offe...</td>
      <td>6</td>
      <td>31</td>
      <td>8</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7202016</td>
      <td>41986501</td>
      <td>2015-08-10</td>
      <td>34959538</td>
      <td>Ming</td>
      <td>Kelly was a great host and very accommodating ...</td>
      <td>0</td>
      <td>33</td>
      <td>8</td>
      <td>2015</td>
    </tr>
  </tbody>
</table>
</div>




```python
def plot_display(plot_words_list):
    wordcloud = WordCloud(width = 1000, height = 700).generate(plot_words_list)
    plt.figure(figsize=(18,12))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
```

We will use Natural Language Toolkit, a Natural Language Processing library for python and use it to process the reviews and comments


```python
from wordcloud import WordCloud
from wordcloud import STOPWORDS

import matplotlib.pyplot as plt
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
comments = df2[['comments','listing_id',]]
commentsDFTopper = comments.sort_values('listing_id',ascending=[0])
commentsDFtop=commentsDFTopper.head(30)
reviews = ''
for index,row in commentsDFtop.iterrows():
    p = re.sub('[^a-zA-Z]+',' ', row['comments'])
    reviews+=p

reviews_data=nltk.word_tokenize(reviews)
filtered_data=[word for word in reviews_data if word not in stopwords.words('english')] 
wnl = nltk.WordNetLemmatizer() 
reviews_data=[wnl.lemmatize(data) for data in filtered_data]
reviews_words=' '.join(reviews_data)
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     /home/fucking/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package punkt to /home/fucking/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package wordnet to /home/fucking/nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!


#### Common words among reviews
Words like great,clean,host, flexible, accomodating were frequently used. These are assocatiated with many listings, hosts should strive to get such


```python
plot_display(reviews)
```


    
![png](output_41_0.png)
    


### Understanding Listings data
We need to look at the data available to us in the listings CSV file, in order to gain understanding of the data.

### Columns
We start with reading in the file and listing out the columns available to us.


```python
listings_df=pd.read_csv("listings.csv", delimiter=',')
listings_df.columns
```




    Index(['id', 'listing_url', 'scrape_id', 'last_scraped', 'name', 'summary',
           'space', 'description', 'experiences_offered', 'neighborhood_overview',
           'notes', 'transit', 'thumbnail_url', 'medium_url', 'picture_url',
           'xl_picture_url', 'host_id', 'host_url', 'host_name', 'host_since',
           'host_location', 'host_about', 'host_response_time',
           'host_response_rate', 'host_acceptance_rate', 'host_is_superhost',
           'host_thumbnail_url', 'host_picture_url', 'host_neighbourhood',
           'host_listings_count', 'host_total_listings_count',
           'host_verifications', 'host_has_profile_pic', 'host_identity_verified',
           'street', 'neighbourhood', 'neighbourhood_cleansed',
           'neighbourhood_group_cleansed', 'city', 'state', 'zipcode', 'market',
           'smart_location', 'country_code', 'country', 'latitude', 'longitude',
           'is_location_exact', 'property_type', 'room_type', 'accommodates',
           'bathrooms', 'bedrooms', 'beds', 'bed_type', 'amenities', 'square_feet',
           'price', 'weekly_price', 'monthly_price', 'security_deposit',
           'cleaning_fee', 'guests_included', 'extra_people', 'minimum_nights',
           'maximum_nights', 'calendar_updated', 'has_availability',
           'availability_30', 'availability_60', 'availability_90',
           'availability_365', 'calendar_last_scraped', 'number_of_reviews',
           'first_review', 'last_review', 'review_scores_rating',
           'review_scores_accuracy', 'review_scores_cleanliness',
           'review_scores_checkin', 'review_scores_communication',
           'review_scores_location', 'review_scores_value', 'requires_license',
           'license', 'jurisdiction_names', 'instant_bookable',
           'cancellation_policy', 'require_guest_profile_picture',
           'require_guest_phone_verification', 'calculated_host_listings_count',
           'reviews_per_month'],
          dtype='object')




```python
# How many data types do we have
listings_df.dtypes.value_counts()
```




    object     62
    float64    17
    int64      13
    dtype: int64




```python
# Integer columns
listings_df.select_dtypes(include = ["int"])[:5]
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
      <th>id</th>
      <th>scrape_id</th>
      <th>host_id</th>
      <th>accommodates</th>
      <th>guests_included</th>
      <th>minimum_nights</th>
      <th>maximum_nights</th>
      <th>availability_30</th>
      <th>availability_60</th>
      <th>availability_90</th>
      <th>availability_365</th>
      <th>number_of_reviews</th>
      <th>calculated_host_listings_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>241032</td>
      <td>20160104002432</td>
      <td>956883</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>365</td>
      <td>14</td>
      <td>41</td>
      <td>71</td>
      <td>346</td>
      <td>207</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>953595</td>
      <td>20160104002432</td>
      <td>5177328</td>
      <td>4</td>
      <td>1</td>
      <td>2</td>
      <td>90</td>
      <td>13</td>
      <td>13</td>
      <td>16</td>
      <td>291</td>
      <td>43</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3308979</td>
      <td>20160104002432</td>
      <td>16708587</td>
      <td>11</td>
      <td>10</td>
      <td>4</td>
      <td>30</td>
      <td>1</td>
      <td>6</td>
      <td>17</td>
      <td>220</td>
      <td>20</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7421966</td>
      <td>20160104002432</td>
      <td>9851441</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1125</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>143</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>278830</td>
      <td>20160104002432</td>
      <td>1452570</td>
      <td>6</td>
      <td>6</td>
      <td>1</td>
      <td>1125</td>
      <td>30</td>
      <td>60</td>
      <td>90</td>
      <td>365</td>
      <td>38</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Float columns
listings_df.select_dtypes(include = ["float"])[:5]
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
      <th>host_listings_count</th>
      <th>host_total_listings_count</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>bathrooms</th>
      <th>bedrooms</th>
      <th>beds</th>
      <th>square_feet</th>
      <th>review_scores_rating</th>
      <th>review_scores_accuracy</th>
      <th>review_scores_cleanliness</th>
      <th>review_scores_checkin</th>
      <th>review_scores_communication</th>
      <th>review_scores_location</th>
      <th>review_scores_value</th>
      <th>license</th>
      <th>reviews_per_month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.0</td>
      <td>3.0</td>
      <td>47.636289</td>
      <td>-122.371025</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>95.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>NaN</td>
      <td>4.07</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6.0</td>
      <td>6.0</td>
      <td>47.639123</td>
      <td>-122.365666</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>96.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>NaN</td>
      <td>1.48</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>2.0</td>
      <td>47.629724</td>
      <td>-122.369483</td>
      <td>4.5</td>
      <td>5.0</td>
      <td>7.0</td>
      <td>NaN</td>
      <td>97.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>NaN</td>
      <td>1.15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>47.638473</td>
      <td>-122.369279</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>2.0</td>
      <td>47.632918</td>
      <td>-122.372471</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>92.0</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>NaN</td>
      <td>0.89</td>
    </tr>
  </tbody>
</table>
</div>




```python
# object columns
listings_df.select_dtypes(include = ["object"])[:5]
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
      <th>listing_url</th>
      <th>last_scraped</th>
      <th>name</th>
      <th>summary</th>
      <th>space</th>
      <th>description</th>
      <th>experiences_offered</th>
      <th>neighborhood_overview</th>
      <th>notes</th>
      <th>transit</th>
      <th>...</th>
      <th>has_availability</th>
      <th>calendar_last_scraped</th>
      <th>first_review</th>
      <th>last_review</th>
      <th>requires_license</th>
      <th>jurisdiction_names</th>
      <th>instant_bookable</th>
      <th>cancellation_policy</th>
      <th>require_guest_profile_picture</th>
      <th>require_guest_phone_verification</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>https://www.airbnb.com/rooms/241032</td>
      <td>2016-01-04</td>
      <td>Stylish Queen Anne Apartment</td>
      <td>NaN</td>
      <td>Make your self at home in this charming one-be...</td>
      <td>Make your self at home in this charming one-be...</td>
      <td>none</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>t</td>
      <td>2016-01-04</td>
      <td>2011-11-01</td>
      <td>2016-01-02</td>
      <td>f</td>
      <td>WASHINGTON</td>
      <td>f</td>
      <td>moderate</td>
      <td>f</td>
      <td>f</td>
    </tr>
    <tr>
      <th>1</th>
      <td>https://www.airbnb.com/rooms/953595</td>
      <td>2016-01-04</td>
      <td>Bright &amp; Airy Queen Anne Apartment</td>
      <td>Chemically sensitive? We've removed the irrita...</td>
      <td>Beautiful, hypoallergenic apartment in an extr...</td>
      <td>Chemically sensitive? We've removed the irrita...</td>
      <td>none</td>
      <td>Queen Anne is a wonderful, truly functional vi...</td>
      <td>What's up with the free pillows?  Our home was...</td>
      <td>Convenient bus stops are just down the block, ...</td>
      <td>...</td>
      <td>t</td>
      <td>2016-01-04</td>
      <td>2013-08-19</td>
      <td>2015-12-29</td>
      <td>f</td>
      <td>WASHINGTON</td>
      <td>f</td>
      <td>strict</td>
      <td>t</td>
      <td>t</td>
    </tr>
    <tr>
      <th>2</th>
      <td>https://www.airbnb.com/rooms/3308979</td>
      <td>2016-01-04</td>
      <td>New Modern House-Amazing water view</td>
      <td>New modern house built in 2013.  Spectacular s...</td>
      <td>Our house is modern, light and fresh with a wa...</td>
      <td>New modern house built in 2013.  Spectacular s...</td>
      <td>none</td>
      <td>Upper Queen Anne is a charming neighborhood fu...</td>
      <td>Our house is located just 5 short blocks to To...</td>
      <td>A bus stop is just 2 blocks away.   Easy bus a...</td>
      <td>...</td>
      <td>t</td>
      <td>2016-01-04</td>
      <td>2014-07-30</td>
      <td>2015-09-03</td>
      <td>f</td>
      <td>WASHINGTON</td>
      <td>f</td>
      <td>strict</td>
      <td>f</td>
      <td>f</td>
    </tr>
    <tr>
      <th>3</th>
      <td>https://www.airbnb.com/rooms/7421966</td>
      <td>2016-01-04</td>
      <td>Queen Anne Chateau</td>
      <td>A charming apartment that sits atop Queen Anne...</td>
      <td>NaN</td>
      <td>A charming apartment that sits atop Queen Anne...</td>
      <td>none</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>t</td>
      <td>2016-01-04</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>f</td>
      <td>WASHINGTON</td>
      <td>f</td>
      <td>flexible</td>
      <td>f</td>
      <td>f</td>
    </tr>
    <tr>
      <th>4</th>
      <td>https://www.airbnb.com/rooms/278830</td>
      <td>2016-01-04</td>
      <td>Charming craftsman 3 bdm house</td>
      <td>Cozy family craftman house in beautiful neighb...</td>
      <td>Cozy family craftman house in beautiful neighb...</td>
      <td>Cozy family craftman house in beautiful neighb...</td>
      <td>none</td>
      <td>We are in the beautiful neighborhood of Queen ...</td>
      <td>Belltown</td>
      <td>The nearest public transit bus (D Line) is 2 b...</td>
      <td>...</td>
      <td>t</td>
      <td>2016-01-04</td>
      <td>2012-07-10</td>
      <td>2015-10-24</td>
      <td>f</td>
      <td>WASHINGTON</td>
      <td>f</td>
      <td>strict</td>
      <td>f</td>
      <td>f</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 62 columns</p>
</div>




```python
#which columns have missing values
listings_df.isnull().sum()[listings_df.isna().sum()>0]
```




    summary                         177
    space                           569
    neighborhood_overview          1032
    notes                          1606
    transit                         934
    thumbnail_url                   320
    medium_url                      320
    xl_picture_url                  320
    host_name                         2
    host_since                        2
    host_location                     8
    host_about                      859
    host_response_time              523
    host_response_rate              523
    host_acceptance_rate            773
    host_is_superhost                 2
    host_thumbnail_url                2
    host_picture_url                  2
    host_neighbourhood              300
    host_listings_count               2
    host_total_listings_count         2
    host_has_profile_pic              2
    host_identity_verified            2
    neighbourhood                   416
    zipcode                           7
    property_type                     1
    bathrooms                        16
    bedrooms                          6
    beds                              1
    square_feet                    3721
    weekly_price                   1809
    monthly_price                  2301
    security_deposit               1952
    cleaning_fee                   1030
    first_review                    627
    last_review                     627
    review_scores_rating            647
    review_scores_accuracy          658
    review_scores_cleanliness       653
    review_scores_checkin           658
    review_scores_communication     651
    review_scores_location          655
    review_scores_value             656
    license                        3818
    reviews_per_month               627
    dtype: int64



The listings data set has 92 features/columns. They are obviously too many and we will need to focus.A quick overview of the lists we realise that there are groupings of features. 

1. Columns about reviews, for instance number_of_reviews,first_review, last_review,review_scores_rating,     review_scores_accuracy, review_scores_cleanliness, review_scores_checkin, review_scores_communication, review_scores_location, review_scores_value
2. There are columns related to property like amenitiess, bedrooms, room type etc
3. There are columns related  to location, properties like neighbourhood,zip cde, city. We will want to see    which neighbourhoods are busiest, pricey
4. We also have columns that are related to the host's policies such as price, weekly pricem security                 deposits 
5. Host related attributes like host location, host response time, these features may be important for user           experience

#### Review based columns


```python
# list of columns related to reviews
review_cols = [col for col in listings_df.columns if 'review' in col]
review_cols
```




    ['number_of_reviews',
     'first_review',
     'last_review',
     'review_scores_rating',
     'review_scores_accuracy',
     'review_scores_cleanliness',
     'review_scores_checkin',
     'review_scores_communication',
     'review_scores_location',
     'review_scores_value',
     'reviews_per_month']




```python
listings_df[review_cols].describe()
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
      <th>number_of_reviews</th>
      <th>review_scores_rating</th>
      <th>review_scores_accuracy</th>
      <th>review_scores_cleanliness</th>
      <th>review_scores_checkin</th>
      <th>review_scores_communication</th>
      <th>review_scores_location</th>
      <th>review_scores_value</th>
      <th>reviews_per_month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3818.000000</td>
      <td>3171.000000</td>
      <td>3160.000000</td>
      <td>3165.000000</td>
      <td>3160.000000</td>
      <td>3167.000000</td>
      <td>3163.000000</td>
      <td>3162.000000</td>
      <td>3191.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>22.223415</td>
      <td>94.539262</td>
      <td>9.636392</td>
      <td>9.556398</td>
      <td>9.786709</td>
      <td>9.809599</td>
      <td>9.608916</td>
      <td>9.452245</td>
      <td>2.078919</td>
    </tr>
    <tr>
      <th>std</th>
      <td>37.730892</td>
      <td>6.606083</td>
      <td>0.698031</td>
      <td>0.797274</td>
      <td>0.595499</td>
      <td>0.568211</td>
      <td>0.629053</td>
      <td>0.750259</td>
      <td>1.822348</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>20.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>0.020000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>93.000000</td>
      <td>9.000000</td>
      <td>9.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>9.000000</td>
      <td>9.000000</td>
      <td>0.695000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>9.000000</td>
      <td>96.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>1.540000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>26.000000</td>
      <td>99.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>474.000000</td>
      <td>100.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>12.150000</td>
    </tr>
  </tbody>
</table>
</div>



There are 3818 rows in total for listings. All columns except for number_of_reviews have missing values. 
Looking at the columns we see some interesting things:

The range between max and min values for number of reviews is 474 which is huge. The mean of 22 is no way close to the median of 9 and the max of 474. 

The standard deviation of the review_scores_rating column at 6.61 is much higher than most other reviewscores columns.

The range of reviews_per_month is a lot smaller than number_of_reviews, with the maximum reviews_per_month at 12.15 compared with the maxium number_of_reviews at 474. This is most likely due to the unbounded time period for which number_of_reviews can be measured, whereas the reviews_per_month is measured over a consistent time period, i.e. a month.


```python
## Whats the correlation between the reviews columns
_df=listings_df[review_cols]
corr=_df.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
```




    <AxesSubplot:>




    
![png](output_54_1.png)
    


Theres strong correlation between basically all the columns except for review scores location.
Review scores checking and review scores communication also have strong correlation.
Another strong correlation is between reviews per month and number of reviews which is understandable since one is subset of the other

One of our data mining objectives is to find out if theres a metric for understanding the quality of guest's experience. The reviews columns can help us answer that.

The review scores rating and seems to be the one that is best suited since the other ones are extensions of and correlated which means they may have been used to calculate the rating.


From our earlier analysis, the review scores rating has 3,171 out of 3,818 values that are not empty for this column, which  means we are missing 17% of our data.


```python
# a look at the review scores rating 
listings_df['review_scores_rating'].hist()
```




    <AxesSubplot:>




    
![png](output_56_1.png)
    


Theres generally high scores for reviews rating skewed positively towards the right which means that reviews scores rating is highly important, getting good reviews is very important for hosts


```python
# Another useful metric to note is the number of reviews vs reviews per month
listings_df[['reviews_per_month','number_of_reviews']].hist()
```




    array([[<AxesSubplot:title={'center':'reviews_per_month'}>,
            <AxesSubplot:title={'center':'number_of_reviews'}>]], dtype=object)




    
![png](output_58_1.png)
    


We already knows theres a strong positive correlation between these two. We notice that, the number of reviews occur within a long range, which just means someone may have many reviews because they have been listed longer. But the reviews per month, shows a more accurate picture


```python
listings_df[['number_of_reviews','first_review','last_review','reviews_per_month']][:5]
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
      <th>number_of_reviews</th>
      <th>first_review</th>
      <th>last_review</th>
      <th>reviews_per_month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>207</td>
      <td>2011-11-01</td>
      <td>2016-01-02</td>
      <td>4.07</td>
    </tr>
    <tr>
      <th>1</th>
      <td>43</td>
      <td>2013-08-19</td>
      <td>2015-12-29</td>
      <td>1.48</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20</td>
      <td>2014-07-30</td>
      <td>2015-09-03</td>
      <td>1.15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>38</td>
      <td>2012-07-10</td>
      <td>2015-10-24</td>
      <td>0.89</td>
    </tr>
  </tbody>
</table>
</div>




```python
listings_df[['review_scores_rating','reviews_per_month']].hist()
```




    array([[<AxesSubplot:title={'center':'review_scores_rating'}>,
            <AxesSubplot:title={'center':'reviews_per_month'}>]], dtype=object)




    
![png](output_61_1.png)
    


So far we see that the reviews per month and review score rating features are useful features for understanding customer experience and thus improving customer experience. As expected some users/hosts have many reviews but fewer reviews per month suggesting they have been listed longer but not neccesarily that they are perfoming better than those with fewer number of ratigs but higher reviews per month

Note: For Data preparation, we may consider a relation between these two features as a response variable for modelling customer experience

#### Price and related features


```python
new_price_col=['price', 'security_deposit', 'cleaning_fee', 'guests_included',
       'extra_people', 'minimum_nights', 'maximum_nights','weekly_price', 'monthly_price']

```

The money columns contain the dollar sign and are of object type. We need to remove the dollar sign and convert the data type to float for quantitative analysis


```python
# We need to remove all $ to the columns 
listings_df['price'] = listings_df['price'].str.extract(r'(\d+)', expand=False)
listings_df['weekly_price'] = listings_df['weekly_price'].str.extract(r'(\d+)', expand=False)
listings_df['monthly_price'] = listings_df['monthly_price'].str.extract(r'(\d+)', expand=False)
listings_df['security_deposit'] = listings_df['security_deposit'].str.extract(r'(\d+)', expand=False)
listings_df['cleaning_fee'] = listings_df['cleaning_fee'].str.extract(r'(\d+)', expand=False)
listings_df['extra_people'] = listings_df['extra_people'].str.extract(r'(\d+)', expand=False)
```


```python
# change the money values to float
listings_df['cleaning_fee'] = listings_df['cleaning_fee'].astype(float, errors = 'raise')
listings_df['security_deposit'] = listings_df['security_deposit'].astype(float, errors = 'raise')
listings_df['extra_people'] = listings_df['extra_people'].astype(float, errors = 'raise')
listings_df['price'] = listings_df['price'].astype(float, errors = 'raise')
listings_df['weekly_price'] = listings_df['weekly_price'].astype(float, errors = 'raise')
listings_df['monthly_price'] = listings_df['monthly_price'].astype(float, errors = 'raise')
```


```python
listings_df[new_price_col][:5]
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
      <th>price</th>
      <th>security_deposit</th>
      <th>cleaning_fee</th>
      <th>guests_included</th>
      <th>extra_people</th>
      <th>minimum_nights</th>
      <th>maximum_nights</th>
      <th>weekly_price</th>
      <th>monthly_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>85.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2</td>
      <td>5.0</td>
      <td>1</td>
      <td>365</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>150.0</td>
      <td>100.0</td>
      <td>40.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>2</td>
      <td>90</td>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>975.0</td>
      <td>1.0</td>
      <td>300.0</td>
      <td>10</td>
      <td>25.0</td>
      <td>4</td>
      <td>30</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>1125</td>
      <td>650.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>450.0</td>
      <td>700.0</td>
      <td>125.0</td>
      <td>6</td>
      <td>15.0</td>
      <td>1</td>
      <td>1125</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# check for missing values in the columns
listings_df[new_price_col].isnull().sum()[listings_df[new_price_col].isnull().sum()>0]
```




    security_deposit    1952
    cleaning_fee        1030
    weekly_price        1809
    monthly_price       2301
    dtype: int64




```python
# What are the summary statics saying
listings_df[new_price_col].describe()
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
      <th>price</th>
      <th>security_deposit</th>
      <th>cleaning_fee</th>
      <th>guests_included</th>
      <th>extra_people</th>
      <th>minimum_nights</th>
      <th>maximum_nights</th>
      <th>weekly_price</th>
      <th>monthly_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3818.00000</td>
      <td>1866.000000</td>
      <td>2788.000000</td>
      <td>3818.000000</td>
      <td>3818.000000</td>
      <td>3818.000000</td>
      <td>3818.000000</td>
      <td>2009.000000</td>
      <td>1517.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>127.71451</td>
      <td>247.292069</td>
      <td>61.710904</td>
      <td>1.672603</td>
      <td>10.667627</td>
      <td>2.369303</td>
      <td>780.447617</td>
      <td>433.651070</td>
      <td>67.986816</td>
    </tr>
    <tr>
      <th>std</th>
      <td>89.16278</td>
      <td>153.486417</td>
      <td>48.830341</td>
      <td>1.311040</td>
      <td>17.585922</td>
      <td>16.305902</td>
      <td>1683.589007</td>
      <td>296.675884</td>
      <td>225.403680</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.00000</td>
      <td>1.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>75.00000</td>
      <td>100.000000</td>
      <td>25.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>60.000000</td>
      <td>232.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>100.00000</td>
      <td>200.000000</td>
      <td>50.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>1125.000000</td>
      <td>476.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>150.00000</td>
      <td>300.000000</td>
      <td>83.000000</td>
      <td>2.000000</td>
      <td>20.000000</td>
      <td>2.000000</td>
      <td>1125.000000</td>
      <td>650.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>999.00000</td>
      <td>995.000000</td>
      <td>300.000000</td>
      <td>15.000000</td>
      <td>300.000000</td>
      <td>1000.000000</td>
      <td>100000.000000</td>
      <td>999.000000</td>
      <td>995.000000</td>
    </tr>
  </tbody>
</table>
</div>



Security deposit has 48% listings, lots of missing values. It has a huge range between the maximum and minimum and the mean. It also has a huge standard deviation. There seems to be different charges that are not standardized perhaps related to other factors like neighbourhood, property type etc. We will invest that further in EDA. Maximum nights have very high values. 75% of listings have up to 20 extra people


```python
fig = plt.figure(figsize = (18,7))
ax = fig.gca()
listings_df[new_price_col].hist(ax=ax)

```

    <ipython-input-52-1b5312bb5443>:3: UserWarning: To output multiple subplots, the figure containing the passed axes is being cleared
      listings_df[new_price_col].hist(ax=ax)





    array([[<AxesSubplot:title={'center':'price'}>,
            <AxesSubplot:title={'center':'security_deposit'}>,
            <AxesSubplot:title={'center':'cleaning_fee'}>],
           [<AxesSubplot:title={'center':'guests_included'}>,
            <AxesSubplot:title={'center':'extra_people'}>,
            <AxesSubplot:title={'center':'minimum_nights'}>],
           [<AxesSubplot:title={'center':'maximum_nights'}>,
            <AxesSubplot:title={'center':'weekly_price'}>,
            <AxesSubplot:title={'center':'monthly_price'}>]], dtype=object)




    
![png](output_72_2.png)
    


It is important while we notice the hist plot to remember that there are some extremely high values (outliers) that may be affecting the distributions. 50% of security deposits are around 200 dollars, Theres a little over 400 listings with security deposits between 0 and 100 which is almost as much as the 50 percent. Of those that charge security deposits,an average of 200 listings charge between 100 dollars to 150 dollars and 300 dollars to 400 dollars. Of those that charge security fees, more than half charge between 0  and 60 dollars. Theres about 1/4 that charge between 60 dollars and 120 dollars. 
On guests included most listings have between 0 and 1 extra guests. Most listings have between 0 and 20 extra people included. 


```python
## Whats the correlation between the money columns
_df2=listings_df[new_price_col]
_corr2=_df2.corr()
sns.heatmap(_corr2, xticklabels=_corr2.columns.values, yticklabels=_corr2.columns.values, annot = True)
```




    <AxesSubplot:>




    
![png](output_74_1.png)
    


Price and cleaning fee have a correlation, likely the cleaning fee is added to the price

#### Location based columns


```python
# look for columns with the word columns in them
location_cols = [col for col in listings_df.columns if 'location' in col]
location_cols
```




    ['host_location',
     'smart_location',
     'is_location_exact',
     'review_scores_location']




```python
listings_df[location_cols][:5]
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
      <th>host_location</th>
      <th>smart_location</th>
      <th>is_location_exact</th>
      <th>review_scores_location</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Seattle, Washington, United States</td>
      <td>Seattle, WA</td>
      <td>t</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Seattle, Washington, United States</td>
      <td>Seattle, WA</td>
      <td>t</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Seattle, Washington, United States</td>
      <td>Seattle, WA</td>
      <td>t</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Seattle, Washington, United States</td>
      <td>Seattle, WA</td>
      <td>t</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Seattle, Washington, United States</td>
      <td>Seattle, WA</td>
      <td>t</td>
      <td>9.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
listings_df[location_cols][:5].dtypes
```




    host_location              object
    smart_location             object
    is_location_exact           int64
    review_scores_location    float64
    dtype: object




```python
# is location exact has T, F for Yes or No, True or false we replace it with True or false

listings_df['is_location_exact']=listings_df['is_location_exact'].replace(to_replace="t",value="1")
listings_df['is_location_exact']=listings_df['is_location_exact'].replace(to_replace="f",value="0")
```


```python
# change is_location_exact from int to Int
listings_df['is_location_exact'] = listings_df['is_location_exact'].astype(int, errors = 'raise')
```


```python
listings_df[location_cols][:5]
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
      <th>host_location</th>
      <th>smart_location</th>
      <th>is_location_exact</th>
      <th>review_scores_location</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Seattle, Washington, United States</td>
      <td>Seattle, WA</td>
      <td>1</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Seattle, Washington, United States</td>
      <td>Seattle, WA</td>
      <td>1</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Seattle, Washington, United States</td>
      <td>Seattle, WA</td>
      <td>1</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Seattle, Washington, United States</td>
      <td>Seattle, WA</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Seattle, Washington, United States</td>
      <td>Seattle, WA</td>
      <td>1</td>
      <td>9.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# are there columns with missing values
listings_df[location_cols].isnull().sum()[listings_df[location_cols].isnull().sum()>0]
```




    host_location               8
    review_scores_location    655
    dtype: int64



The review scores location has the key word location in it but has been considered more relevantly under the review based columns so we will drop it. Also, upon rechecking the listings data see there are other columns that deal with location but dont have location in their column names. We include these here along with the location cols


```python
location_cols=['city','state','zipcode','market','smart_location','country_code','country','host_location','is_location_exact']
```


```python
listings_df[location_cols][:5]
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
      <th>city</th>
      <th>state</th>
      <th>zipcode</th>
      <th>market</th>
      <th>smart_location</th>
      <th>country_code</th>
      <th>country</th>
      <th>host_location</th>
      <th>is_location_exact</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Seattle</td>
      <td>WA</td>
      <td>98119</td>
      <td>Seattle</td>
      <td>Seattle, WA</td>
      <td>US</td>
      <td>United States</td>
      <td>Seattle, Washington, United States</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Seattle</td>
      <td>WA</td>
      <td>98119</td>
      <td>Seattle</td>
      <td>Seattle, WA</td>
      <td>US</td>
      <td>United States</td>
      <td>Seattle, Washington, United States</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Seattle</td>
      <td>WA</td>
      <td>98119</td>
      <td>Seattle</td>
      <td>Seattle, WA</td>
      <td>US</td>
      <td>United States</td>
      <td>Seattle, Washington, United States</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Seattle</td>
      <td>WA</td>
      <td>98119</td>
      <td>Seattle</td>
      <td>Seattle, WA</td>
      <td>US</td>
      <td>United States</td>
      <td>Seattle, Washington, United States</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Seattle</td>
      <td>WA</td>
      <td>98119</td>
      <td>Seattle</td>
      <td>Seattle, WA</td>
      <td>US</td>
      <td>United States</td>
      <td>Seattle, Washington, United States</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#exclude the is_location exact and find some summary stats
listings_df[['city','state','zipcode','market','smart_location','country_code','country']].describe()
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
      <th>city</th>
      <th>state</th>
      <th>zipcode</th>
      <th>market</th>
      <th>smart_location</th>
      <th>country_code</th>
      <th>country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3818</td>
      <td>3818</td>
      <td>3811</td>
      <td>3818</td>
      <td>3818</td>
      <td>3818</td>
      <td>3818</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>7</td>
      <td>2</td>
      <td>28</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>top</th>
      <td>Seattle</td>
      <td>WA</td>
      <td>98122</td>
      <td>Seattle</td>
      <td>Seattle, WA</td>
      <td>US</td>
      <td>United States</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>3810</td>
      <td>3817</td>
      <td>420</td>
      <td>3818</td>
      <td>3810</td>
      <td>3818</td>
      <td>3818</td>
    </tr>
  </tbody>
</table>
</div>



Zip code has the most count of unique entries with 27 different entries, city has 7 unique entries, the rest are probably the same for instance, same market, country code and country. So we exclude country, country code and state to see what we can learn. Almost all listings are in Seattle, WA thats where we have  99.9% of listings, other areas have 1%


```python
listings_df[['city','state','zipcode','smart_location']].groupby(['city']).count()
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
      <th>state</th>
      <th>zipcode</th>
      <th>smart_location</th>
    </tr>
    <tr>
      <th>city</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ballard, Seattle</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Phinney Ridge Seattle</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Seattle</th>
      <td>3810</td>
      <td>3803</td>
      <td>3810</td>
    </tr>
    <tr>
      <th>Seattle</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>West Seattle</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>seattle</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>西雅图</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
(listings_df['zipcode'].value_counts()).plot(kind="bar")
```




    <AxesSubplot:>




    
![png](output_90_1.png)
    


We see that there are zipcodes that have more listings like the 98122, 98103 and 98102. These are generally higher, the bulk of the listings are on average the same until we get to 98104. Then we see 98199, 126,106,108,133,136 have about the same listings around the 50-60 listings average

### Host attributes columns


```python
host_cols = [col for col in listings_df.columns if 'host' in col]
host_cols
```




    ['host_id',
     'host_url',
     'host_name',
     'host_since',
     'host_location',
     'host_about',
     'host_response_time',
     'host_response_rate',
     'host_acceptance_rate',
     'host_is_superhost',
     'host_thumbnail_url',
     'host_picture_url',
     'host_neighbourhood',
     'host_listings_count',
     'host_total_listings_count',
     'host_verifications',
     'host_has_profile_pic',
     'host_identity_verified',
     'calculated_host_listings_count']




```python
listings_df[host_cols][:5]
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
      <th>host_id</th>
      <th>host_url</th>
      <th>host_name</th>
      <th>host_since</th>
      <th>host_location</th>
      <th>host_about</th>
      <th>host_response_time</th>
      <th>host_response_rate</th>
      <th>host_acceptance_rate</th>
      <th>host_is_superhost</th>
      <th>host_thumbnail_url</th>
      <th>host_picture_url</th>
      <th>host_neighbourhood</th>
      <th>host_listings_count</th>
      <th>host_total_listings_count</th>
      <th>host_verifications</th>
      <th>host_has_profile_pic</th>
      <th>host_identity_verified</th>
      <th>calculated_host_listings_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>956883</td>
      <td>https://www.airbnb.com/users/show/956883</td>
      <td>Maija</td>
      <td>2011-08-11</td>
      <td>Seattle, Washington, United States</td>
      <td>I am an artist, interior designer, and run a s...</td>
      <td>within a few hours</td>
      <td>96%</td>
      <td>100%</td>
      <td>f</td>
      <td>https://a0.muscache.com/ac/users/956883/profil...</td>
      <td>https://a0.muscache.com/ac/users/956883/profil...</td>
      <td>Queen Anne</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>['email', 'phone', 'reviews', 'kba']</td>
      <td>t</td>
      <td>t</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5177328</td>
      <td>https://www.airbnb.com/users/show/5177328</td>
      <td>Andrea</td>
      <td>2013-02-21</td>
      <td>Seattle, Washington, United States</td>
      <td>Living east coast/left coast/overseas.  Time i...</td>
      <td>within an hour</td>
      <td>98%</td>
      <td>100%</td>
      <td>t</td>
      <td>https://a0.muscache.com/ac/users/5177328/profi...</td>
      <td>https://a0.muscache.com/ac/users/5177328/profi...</td>
      <td>Queen Anne</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>['email', 'phone', 'facebook', 'linkedin', 're...</td>
      <td>t</td>
      <td>t</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>16708587</td>
      <td>https://www.airbnb.com/users/show/16708587</td>
      <td>Jill</td>
      <td>2014-06-12</td>
      <td>Seattle, Washington, United States</td>
      <td>i love living in Seattle.  i grew up in the mi...</td>
      <td>within a few hours</td>
      <td>67%</td>
      <td>100%</td>
      <td>f</td>
      <td>https://a1.muscache.com/ac/users/16708587/prof...</td>
      <td>https://a1.muscache.com/ac/users/16708587/prof...</td>
      <td>Queen Anne</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>['email', 'phone', 'google', 'reviews', 'jumio']</td>
      <td>t</td>
      <td>t</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9851441</td>
      <td>https://www.airbnb.com/users/show/9851441</td>
      <td>Emily</td>
      <td>2013-11-06</td>
      <td>Seattle, Washington, United States</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>f</td>
      <td>https://a2.muscache.com/ac/users/9851441/profi...</td>
      <td>https://a2.muscache.com/ac/users/9851441/profi...</td>
      <td>Queen Anne</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>['email', 'phone', 'facebook', 'reviews', 'jum...</td>
      <td>t</td>
      <td>t</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1452570</td>
      <td>https://www.airbnb.com/users/show/1452570</td>
      <td>Emily</td>
      <td>2011-11-29</td>
      <td>Seattle, Washington, United States</td>
      <td>Hi, I live in Seattle, Washington but I'm orig...</td>
      <td>within an hour</td>
      <td>100%</td>
      <td>NaN</td>
      <td>f</td>
      <td>https://a0.muscache.com/ac/users/1452570/profi...</td>
      <td>https://a0.muscache.com/ac/users/1452570/profi...</td>
      <td>Queen Anne</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>['email', 'phone', 'facebook', 'reviews', 'kba']</td>
      <td>t</td>
      <td>t</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#create a host listings df from the listings,a df that contains only listing data
host_listings_df=listings_df[host_cols].copy()
```


```python
# find some correlations on the host listing columns
host_listings_df.corr()
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
      <th>host_id</th>
      <th>host_listings_count</th>
      <th>host_total_listings_count</th>
      <th>calculated_host_listings_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>host_id</th>
      <td>1.000000</td>
      <td>-0.069613</td>
      <td>-0.069613</td>
      <td>-0.176040</td>
    </tr>
    <tr>
      <th>host_listings_count</th>
      <td>-0.069613</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.224222</td>
    </tr>
    <tr>
      <th>host_total_listings_count</th>
      <td>-0.069613</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.224222</td>
    </tr>
    <tr>
      <th>calculated_host_listings_count</th>
      <td>-0.176040</td>
      <td>0.224222</td>
      <td>0.224222</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



The corr( ) function returns 4 features, host_id, listings_count, total_listings_count and calculated host listings count. There is no description from AirBnb on what these columns are. We see strong positive 100 percent correlation between host listings  count and host total listings count and no other correlation. These two might be the same thing


```python
#host_listings_df
pct_to_float = lambda x: x.str.replace(r'%', r'.0').astype('float') / 100.0

# note that the percentages are now in float e.g 96% is 0.96
# Apply the function to the rate cols

host_listings_df[['host_response_rate','host_acceptance_rate']] = host_listings_df[['host_response_rate','host_acceptance_rate']].apply(pct_to_float, axis=1)
```


```python
host_listings_df[:5]
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
      <th>host_id</th>
      <th>host_url</th>
      <th>host_name</th>
      <th>host_since</th>
      <th>host_location</th>
      <th>host_about</th>
      <th>host_response_time</th>
      <th>host_response_rate</th>
      <th>host_acceptance_rate</th>
      <th>host_is_superhost</th>
      <th>host_thumbnail_url</th>
      <th>host_picture_url</th>
      <th>host_neighbourhood</th>
      <th>host_listings_count</th>
      <th>host_total_listings_count</th>
      <th>host_verifications</th>
      <th>host_has_profile_pic</th>
      <th>host_identity_verified</th>
      <th>calculated_host_listings_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>956883</td>
      <td>https://www.airbnb.com/users/show/956883</td>
      <td>Maija</td>
      <td>2011-08-11</td>
      <td>Seattle, Washington, United States</td>
      <td>I am an artist, interior designer, and run a s...</td>
      <td>within a few hours</td>
      <td>0.96</td>
      <td>1.0</td>
      <td>f</td>
      <td>https://a0.muscache.com/ac/users/956883/profil...</td>
      <td>https://a0.muscache.com/ac/users/956883/profil...</td>
      <td>Queen Anne</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>['email', 'phone', 'reviews', 'kba']</td>
      <td>t</td>
      <td>t</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5177328</td>
      <td>https://www.airbnb.com/users/show/5177328</td>
      <td>Andrea</td>
      <td>2013-02-21</td>
      <td>Seattle, Washington, United States</td>
      <td>Living east coast/left coast/overseas.  Time i...</td>
      <td>within an hour</td>
      <td>0.98</td>
      <td>1.0</td>
      <td>t</td>
      <td>https://a0.muscache.com/ac/users/5177328/profi...</td>
      <td>https://a0.muscache.com/ac/users/5177328/profi...</td>
      <td>Queen Anne</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>['email', 'phone', 'facebook', 'linkedin', 're...</td>
      <td>t</td>
      <td>t</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>16708587</td>
      <td>https://www.airbnb.com/users/show/16708587</td>
      <td>Jill</td>
      <td>2014-06-12</td>
      <td>Seattle, Washington, United States</td>
      <td>i love living in Seattle.  i grew up in the mi...</td>
      <td>within a few hours</td>
      <td>0.67</td>
      <td>1.0</td>
      <td>f</td>
      <td>https://a1.muscache.com/ac/users/16708587/prof...</td>
      <td>https://a1.muscache.com/ac/users/16708587/prof...</td>
      <td>Queen Anne</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>['email', 'phone', 'google', 'reviews', 'jumio']</td>
      <td>t</td>
      <td>t</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9851441</td>
      <td>https://www.airbnb.com/users/show/9851441</td>
      <td>Emily</td>
      <td>2013-11-06</td>
      <td>Seattle, Washington, United States</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>f</td>
      <td>https://a2.muscache.com/ac/users/9851441/profi...</td>
      <td>https://a2.muscache.com/ac/users/9851441/profi...</td>
      <td>Queen Anne</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>['email', 'phone', 'facebook', 'reviews', 'jum...</td>
      <td>t</td>
      <td>t</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1452570</td>
      <td>https://www.airbnb.com/users/show/1452570</td>
      <td>Emily</td>
      <td>2011-11-29</td>
      <td>Seattle, Washington, United States</td>
      <td>Hi, I live in Seattle, Washington but I'm orig...</td>
      <td>within an hour</td>
      <td>1.00</td>
      <td>NaN</td>
      <td>f</td>
      <td>https://a0.muscache.com/ac/users/1452570/profi...</td>
      <td>https://a0.muscache.com/ac/users/1452570/profi...</td>
      <td>Queen Anne</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>['email', 'phone', 'facebook', 'reviews', 'kba']</td>
      <td>t</td>
      <td>t</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



We beging to investigate the **host listings**, theres a number of features, we investigate them in smaller batches starting with the rates


```python
host_listings_df[['host_response_rate','host_acceptance_rate']].hist()
```




    array([[<AxesSubplot:title={'center':'host_response_rate'}>,
            <AxesSubplot:title={'center':'host_acceptance_rate'}>]],
          dtype=object)




    
![png](output_101_1.png)
    


host acceptance rate is always 100% in all occurences, there are no rejections from hosts, host response rate is also almost always 100%, very few hosts have less than 80% response rate

One of the data mining questions is about the **increase in listings in AirBnb**, we look at the host since feature


```python
# convert it to date time
host_listings_df['host_since']=pd.to_datetime(host_listings_df['host_since'])

```


```python
# split the date time into week,month and year
host_listings_df['dayofweek'] = host_listings_df['host_since'].dt.dayofweek
host_listings_df['weekofyear'] = host_listings_df['host_since'].dt.isocalendar().week
host_listings_df['month'] = host_listings_df['host_since'].dt.month
host_listings_df['year'] = host_listings_df['host_since'].dt.year
```


```python
host_listings_df[:5]
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
      <th>host_id</th>
      <th>host_url</th>
      <th>host_name</th>
      <th>host_since</th>
      <th>host_location</th>
      <th>host_about</th>
      <th>host_response_time</th>
      <th>host_response_rate</th>
      <th>host_acceptance_rate</th>
      <th>host_is_superhost</th>
      <th>...</th>
      <th>host_listings_count</th>
      <th>host_total_listings_count</th>
      <th>host_verifications</th>
      <th>host_has_profile_pic</th>
      <th>host_identity_verified</th>
      <th>calculated_host_listings_count</th>
      <th>dayofweek</th>
      <th>weekofyear</th>
      <th>month</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>956883</td>
      <td>https://www.airbnb.com/users/show/956883</td>
      <td>Maija</td>
      <td>2011-08-11</td>
      <td>Seattle, Washington, United States</td>
      <td>I am an artist, interior designer, and run a s...</td>
      <td>within a few hours</td>
      <td>0.96</td>
      <td>1.0</td>
      <td>f</td>
      <td>...</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>['email', 'phone', 'reviews', 'kba']</td>
      <td>t</td>
      <td>t</td>
      <td>2</td>
      <td>3.0</td>
      <td>32</td>
      <td>8.0</td>
      <td>2011.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5177328</td>
      <td>https://www.airbnb.com/users/show/5177328</td>
      <td>Andrea</td>
      <td>2013-02-21</td>
      <td>Seattle, Washington, United States</td>
      <td>Living east coast/left coast/overseas.  Time i...</td>
      <td>within an hour</td>
      <td>0.98</td>
      <td>1.0</td>
      <td>t</td>
      <td>...</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>['email', 'phone', 'facebook', 'linkedin', 're...</td>
      <td>t</td>
      <td>t</td>
      <td>6</td>
      <td>3.0</td>
      <td>8</td>
      <td>2.0</td>
      <td>2013.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>16708587</td>
      <td>https://www.airbnb.com/users/show/16708587</td>
      <td>Jill</td>
      <td>2014-06-12</td>
      <td>Seattle, Washington, United States</td>
      <td>i love living in Seattle.  i grew up in the mi...</td>
      <td>within a few hours</td>
      <td>0.67</td>
      <td>1.0</td>
      <td>f</td>
      <td>...</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>['email', 'phone', 'google', 'reviews', 'jumio']</td>
      <td>t</td>
      <td>t</td>
      <td>2</td>
      <td>3.0</td>
      <td>24</td>
      <td>6.0</td>
      <td>2014.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9851441</td>
      <td>https://www.airbnb.com/users/show/9851441</td>
      <td>Emily</td>
      <td>2013-11-06</td>
      <td>Seattle, Washington, United States</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>f</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>['email', 'phone', 'facebook', 'reviews', 'jum...</td>
      <td>t</td>
      <td>t</td>
      <td>1</td>
      <td>2.0</td>
      <td>45</td>
      <td>11.0</td>
      <td>2013.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1452570</td>
      <td>https://www.airbnb.com/users/show/1452570</td>
      <td>Emily</td>
      <td>2011-11-29</td>
      <td>Seattle, Washington, United States</td>
      <td>Hi, I live in Seattle, Washington but I'm orig...</td>
      <td>within an hour</td>
      <td>1.00</td>
      <td>NaN</td>
      <td>f</td>
      <td>...</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>['email', 'phone', 'facebook', 'reviews', 'kba']</td>
      <td>t</td>
      <td>t</td>
      <td>1</td>
      <td>1.0</td>
      <td>48</td>
      <td>11.0</td>
      <td>2011.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>



We want to see the statistics of which days of the week, week of the year occur more and what those trends are


```python
host_listings_df[['host_since','dayofweek','weekofyear','month','year']].describe()
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
      <th>dayofweek</th>
      <th>weekofyear</th>
      <th>month</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3816.000000</td>
      <td>3816.000000</td>
      <td>3816.000000</td>
      <td>3816.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.634696</td>
      <td>26.816038</td>
      <td>6.551887</td>
      <td>2013.214623</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.013244</td>
      <td>13.972345</td>
      <td>3.208349</td>
      <td>1.560423</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2008.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>15.000000</td>
      <td>4.000000</td>
      <td>2012.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.000000</td>
      <td>28.000000</td>
      <td>7.000000</td>
      <td>2013.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.000000</td>
      <td>38.000000</td>
      <td>9.000000</td>
      <td>2015.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>6.000000</td>
      <td>53.000000</td>
      <td>12.000000</td>
      <td>2016.000000</td>
    </tr>
  </tbody>
</table>
</div>



On average, about half of the hosts have been hosts since July 2013 and interestingly joined on a Wednesday. The maximum year is 2016 because this dataset was compiled in 2016.


```python
host_listings_df[['host_since','dayofweek','weekofyear','month']].hist()
```




    array([[<AxesSubplot:title={'center':'dayofweek'}>,
            <AxesSubplot:title={'center':'weekofyear'}>],
           [<AxesSubplot:title={'center':'month'}>, <AxesSubplot:>]],
          dtype=object)




    
![png](output_110_1.png)
    


The highest number of hosts joined on the first day of the week and second day of the week (According to google,in the US thats Sunday and Monday). The distribution of the months is spread accross the whole year, with the most joining between weeks 26 and 38. The months distribution provides us an even clearer picture showing highest joins beginning, mid and end of the year.


```python
host_listings_df['host_since'].hist()
```




    <AxesSubplot:>




    
![png](output_112_1.png)
    


This answers one of our data mining questions on whether there has been an increase in hosts joining AirBnb. **Since 2008 when AirBnb started, there has been a huge growth**, with each year there's at least an increament of 100. Airbnb is experiencing significant growth.


```python
host_listings_df[['host_since','year','host_neighbourhood']].groupby(['year']).count()
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
      <th>host_since</th>
      <th>host_neighbourhood</th>
    </tr>
    <tr>
      <th>year</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2008.0</th>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2009.0</th>
      <td>64</td>
      <td>63</td>
    </tr>
    <tr>
      <th>2010.0</th>
      <td>149</td>
      <td>146</td>
    </tr>
    <tr>
      <th>2011.0</th>
      <td>398</td>
      <td>389</td>
    </tr>
    <tr>
      <th>2012.0</th>
      <td>539</td>
      <td>514</td>
    </tr>
    <tr>
      <th>2013.0</th>
      <td>776</td>
      <td>723</td>
    </tr>
    <tr>
      <th>2014.0</th>
      <td>900</td>
      <td>809</td>
    </tr>
    <tr>
      <th>2015.0</th>
      <td>981</td>
      <td>866</td>
    </tr>
    <tr>
      <th>2016.0</th>
      <td>5</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



This further elaborates our findings that there's an upward trend of AirBnB listings since 2008 to 2016. In 10 years, there was a growth from 4 listings to 3816 listings, the growth is exponential. Between 2009 and 2011 the growth rate doubled at a average growth rate of 140% per year. From 2011 the growth rate steadied to an average growth rate of 30%. 2016 has a suprisingly low number possibly due to when this data was collected. We will investigate this. Otherwise, between 2008 and 2016, we see AirBnb listings grow significantly.


```python
host_listings_df['host_since'].describe(datetime_is_numeric=True)
```




    count                             3816
    mean     2013-09-18 17:26:02.264153856
    min                2008-11-10 00:00:00
    25%                2012-08-17 00:00:00
    50%                2013-12-12 12:00:00
    75%                2015-01-14 00:00:00
    max                2016-01-03 00:00:00
    Name: host_since, dtype: object



An investigation into the host since column reveals why 2016 has such a low count for listings. The dataset was collected 03-01-2016, on just 3 days of January there was already 4 listings at 1.33 listings per day.


```python
listings_df[['host_since','neighbourhood_cleansed']].describe()
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
      <th>host_since</th>
      <th>neighbourhood_cleansed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3816</td>
      <td>3818</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>1380</td>
      <td>87</td>
    </tr>
    <tr>
      <th>top</th>
      <td>2013-08-30</td>
      <td>Broadway</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>51</td>
      <td>397</td>
    </tr>
  </tbody>
</table>
</div>



There are 87 neighbourhoods listed. Broadway is the most frequently listed neighbourhood which occurs 397 times


```python
host_listings_df['host_is_superhost'].value_counts()/host_listings_df.shape[0]
```




    f    0.795705
    t    0.203772
    Name: host_is_superhost, dtype: float64



For 70% of the listings the host is not the superhost


```python
host_listings_df['host_has_profile_pic'].value_counts()/host_listings_df.shape[0]
```




    t    0.997643
    f    0.001833
    Name: host_has_profile_pic, dtype: float64



Every host has a profile picture, probably only 1 host in the entire listing doesnt have a picture


```python
host_listings_df['host_identity_verified'].value_counts()/host_listings_df.shape[0]
```




    t    0.784966
    f    0.214510
    Name: host_identity_verified, dtype: float64



Most host's identities are verified, only 21.4% of hosts aren't verified


```python
host_listings_df['host_verifications'].describe()
```




    count                                                 3818
    unique                                                 116
    top       ['email', 'phone', 'facebook', 'reviews', 'kba']
    freq                                                   595
    Name: host_verifications, dtype: object



There are as many host verifications as there are listings. I assume since its unspecified that host verifications are information about hosts that are given out by the hosts to verify. Most hosts provide email, phone, facebook, reviews and kba for verification


```python
host_listings_df['host_response_time'].value_counts()/host_listings_df.shape[0]
```




    within an hour        0.443164
    within a few hours    0.253536
    within a day          0.156365
    a few days or more    0.009953
    Name: host_response_time, dtype: float64



44% of hosts respond within an hour, while 25% respond within a few hours, meaning that a total of 69% of hosts respond well in time guests. 15% will respond within a day. Basically hosts respond at most within a day



```python
host_listings_df['host_thumbnail_url'].describe()
```




    count                                                  3816
    unique                                                 2743
    top       https://a2.muscache.com/ac/pictures/a4d7d053-c...
    freq                                                     46
    Name: host_thumbnail_url, dtype: object




```python
host_listings_df['host_url'].describe()
```




    count                                          3818
    unique                                         2751
    top       https://www.airbnb.com/users/show/8534462
    freq                                             46
    Name: host_url, dtype: object



The same is true of host urls, every host has a url


```python
host_listings_df['host_name'].describe()
```




    count       3816
    unique      1466
    top       Andrew
    freq          56
    Name: host_name, dtype: object




```python
host_listings_df['host_about'].describe()

```




    count                                                  2959
    unique                                                 2011
    top       It would be my pleasure to share and explore t...
    freq                                                     46
    Name: host_about, dtype: object



Every host has a thumbnail, they aren't very unique.The most frequently used one is shown


```python
host_listings_df['host_neighbourhood'].describe()
```




    count             3518
    unique             102
    top       Capitol Hill
    freq               405
    Name: host_neighbourhood, dtype: object



Most hosts stay in capitol Hill


```python
count_vals = host_listings_df.host_neighbourhood.value_counts()
(count_vals[:20]/host_listings_df.shape[0]).plot(kind="bar");
plt.title("Neighbourhood ");
```


    
![png](output_138_0.png)
    


### Policies columns

Here we look at columns that related to host policy like 'requires_license', 'license', 'jurisdiction_names', instant_bookable', 'cancellation_policy', 'require_guest_profile_picture',  'require_guest_phone_verification'


       


```python
policies=['requires_license', 'license', 'jurisdiction_names', 'instant_bookable', 'cancellation_policy','experiences_offered', 'require_guest_profile_picture',  'require_guest_phone_verification']
```


```python
policies_df=listings_df[policies]
```


```python
policies_df[:5]
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
      <th>requires_license</th>
      <th>license</th>
      <th>jurisdiction_names</th>
      <th>instant_bookable</th>
      <th>cancellation_policy</th>
      <th>experiences_offered</th>
      <th>require_guest_profile_picture</th>
      <th>require_guest_phone_verification</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>f</td>
      <td>NaN</td>
      <td>WASHINGTON</td>
      <td>f</td>
      <td>moderate</td>
      <td>none</td>
      <td>f</td>
      <td>f</td>
    </tr>
    <tr>
      <th>1</th>
      <td>f</td>
      <td>NaN</td>
      <td>WASHINGTON</td>
      <td>f</td>
      <td>strict</td>
      <td>none</td>
      <td>t</td>
      <td>t</td>
    </tr>
    <tr>
      <th>2</th>
      <td>f</td>
      <td>NaN</td>
      <td>WASHINGTON</td>
      <td>f</td>
      <td>strict</td>
      <td>none</td>
      <td>f</td>
      <td>f</td>
    </tr>
    <tr>
      <th>3</th>
      <td>f</td>
      <td>NaN</td>
      <td>WASHINGTON</td>
      <td>f</td>
      <td>flexible</td>
      <td>none</td>
      <td>f</td>
      <td>f</td>
    </tr>
    <tr>
      <th>4</th>
      <td>f</td>
      <td>NaN</td>
      <td>WASHINGTON</td>
      <td>f</td>
      <td>strict</td>
      <td>none</td>
      <td>f</td>
      <td>f</td>
    </tr>
  </tbody>
</table>
</div>




```python
policies_df['requires_license'].value_counts()/policies_df.shape[0]
```




    f    1.0
    Name: requires_license, dtype: float64



All of the listings do not require license


```python
policies_df['require_guest_phone_verification'].value_counts()/policies_df.shape[0]
```




    f    0.901781
    t    0.098219
    Name: require_guest_phone_verification, dtype: float64



90% of hosts do not require guest phone verification


```python
policies_df['require_guest_profile_picture'].value_counts()/policies_df.shape[0]
```




    f    0.915925
    t    0.084075
    Name: require_guest_profile_picture, dtype: float64



91% of hosts do not require guest profile picture


```python
policies_df['cancellation_policy'].value_counts()/policies_df.shape[0]
```




    strict      0.371137
    moderate    0.327658
    flexible    0.301205
    Name: cancellation_policy, dtype: float64



On cancelation policy, hosts are split almost evenly between strict, moderate and flexible. 37% being very strict on cancelation


```python
policies_df['instant_bookable'].value_counts()/policies_df.shape[0]
```




    f    0.845207
    t    0.154793
    Name: instant_bookable, dtype: float64



About 84.5% of listings are not instant bookable


```python
policies_df['experiences_offered'].value_counts()
```




    none    3818
    Name: experiences_offered, dtype: int64




```python
available=['has_availability',
       'availability_30', 'availability_60', 'availability_90',
       'availability_365']
```


```python
available_df=listings_df[available]
```


```python
available_df[:5]
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
      <th>has_availability</th>
      <th>availability_30</th>
      <th>availability_60</th>
      <th>availability_90</th>
      <th>availability_365</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>t</td>
      <td>14</td>
      <td>41</td>
      <td>71</td>
      <td>346</td>
    </tr>
    <tr>
      <th>1</th>
      <td>t</td>
      <td>13</td>
      <td>13</td>
      <td>16</td>
      <td>291</td>
    </tr>
    <tr>
      <th>2</th>
      <td>t</td>
      <td>1</td>
      <td>6</td>
      <td>17</td>
      <td>220</td>
    </tr>
    <tr>
      <th>3</th>
      <td>t</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>143</td>
    </tr>
    <tr>
      <th>4</th>
      <td>t</td>
      <td>30</td>
      <td>60</td>
      <td>90</td>
      <td>365</td>
    </tr>
  </tbody>
</table>
</div>




```python
available_df['has_availability'].value_counts()
```




    t    3818
    Name: has_availability, dtype: int64



All have availability. All listings have availability in general. We just have to inspect how much availability they have


```python
available_df[['availability_30', 'availability_60', 'availability_90', 'availability_365']].describe()
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
      <th>availability_30</th>
      <th>availability_60</th>
      <th>availability_90</th>
      <th>availability_365</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3818.000000</td>
      <td>3818.000000</td>
      <td>3818.000000</td>
      <td>3818.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>16.786276</td>
      <td>36.814825</td>
      <td>58.082504</td>
      <td>244.772656</td>
    </tr>
    <tr>
      <th>std</th>
      <td>12.173637</td>
      <td>23.337541</td>
      <td>34.063845</td>
      <td>126.772526</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>13.000000</td>
      <td>28.000000</td>
      <td>124.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>20.000000</td>
      <td>46.000000</td>
      <td>73.000000</td>
      <td>308.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>30.000000</td>
      <td>59.000000</td>
      <td>89.000000</td>
      <td>360.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>30.000000</td>
      <td>60.000000</td>
      <td>90.000000</td>
      <td>365.000000</td>
    </tr>
  </tbody>
</table>
</div>



On average, about 17 houses have availability 30 days, 37 have availability 60 days a year, 58 for 90 days a year and about 245 have availability 365 days of the year.

### Property related columns


```python
property_=['id','property_type', 'room_type', 'accommodates',
       'bathrooms', 'bedrooms', 'beds', 'bed_type','amenities']
```


```python
property_df=listings_df[property_]
```


```python
property_df[:5]
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
      <th>id</th>
      <th>property_type</th>
      <th>room_type</th>
      <th>accommodates</th>
      <th>bathrooms</th>
      <th>bedrooms</th>
      <th>beds</th>
      <th>bed_type</th>
      <th>amenities</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>241032</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>4</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>{TV,"Cable TV",Internet,"Wireless Internet","A...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>953595</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>4</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>{TV,Internet,"Wireless Internet",Kitchen,"Free...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3308979</td>
      <td>House</td>
      <td>Entire home/apt</td>
      <td>11</td>
      <td>4.5</td>
      <td>5.0</td>
      <td>7.0</td>
      <td>Real Bed</td>
      <td>{TV,"Cable TV",Internet,"Wireless Internet","A...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7421966</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>3</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>Real Bed</td>
      <td>{Internet,"Wireless Internet",Kitchen,"Indoor ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>278830</td>
      <td>House</td>
      <td>Entire home/apt</td>
      <td>6</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>Real Bed</td>
      <td>{TV,"Cable TV",Internet,"Wireless Internet",Ki...</td>
    </tr>
  </tbody>
</table>
</div>




```python
property_df.groupby(['room_type']).id.count()/property_df.shape[0]
```




    room_type
    Entire home/apt    0.665532
    Private room       0.303824
    Shared room        0.030644
    Name: id, dtype: float64




```python
(property_df.groupby(['room_type']).id.count()/property_df.shape[0]).plot(kind="pie")
```




    <AxesSubplot:ylabel='id'>




    
![png](output_167_1.png)
    


66.5% of properties are are entire homes/appartments. 30% are private rooms, only 3% of listings are shared rooms


```python
property_df.groupby(['property_type']).id.count()/property_df.shape[0]
```




    property_type
    Apartment          0.447355
    Bed & Breakfast    0.009691
    Boat               0.002095
    Bungalow           0.003405
    Cabin              0.005500
    Camper/RV          0.003405
    Chalet             0.000524
    Condominium        0.023834
    Dorm               0.000524
    House              0.453903
    Loft               0.010477
    Other              0.005762
    Tent               0.001310
    Townhouse          0.030906
    Treehouse          0.000786
    Yurt               0.000262
    Name: id, dtype: float64




```python
(property_df.groupby(['property_type']).id.count()/property_df.shape[0]).plot(kind="bar")
```




    <AxesSubplot:xlabel='property_type'>




    
![png](output_170_1.png)
    


Appartments and houses are the most common properties with 45% and 44.7% respectively. Bed and breakfast at 95 and the rest shared between other property types


```python
property_df.groupby(['bed_type']).id.count()/property_df.shape[0]
```




    bed_type
    Airbed           0.007072
    Couch            0.003405
    Futon            0.019382
    Pull-out Sofa    0.012310
    Real Bed         0.957831
    Name: id, dtype: float64




```python
(property_df.groupby(['bed_type']).id.count()/property_df.shape[0]).plot(kind="bar")
```




    <AxesSubplot:xlabel='bed_type'>




    
![png](output_173_1.png)
    


95% of beds listed are real beds, about 2% being Futons and the rest not beds


```python
property_df.groupby(['accommodates']).id.count()/property_df.shape[0]
```




    accommodates
    1     0.067051
    2     0.426139
    3     0.104243
    4     0.205605
    5     0.048193
    6     0.086957
    7     0.013620
    8     0.031168
    9     0.003405
    10    0.006548
    11    0.000786
    12    0.003929
    14    0.000786
    15    0.000524
    16    0.001048
    Name: id, dtype: float64




```python
(property_df.groupby(['accommodates']).id.count()/property_df.shape[0]).plot(kind="bar")
```




    <AxesSubplot:xlabel='accommodates'>




    
![png](output_176_1.png)
    


42% of listings accomodate a maximum of 2 guests, 20% accomodate 4, 10% accomodate 3, smaller percentages can accomodate more than 4 people. This makes sense since most listings are houses and apartments


```python
property_df.groupby(['bedrooms']).id.count()/property_df.shape[0]
```




    bedrooms
    0.0    0.097433
    1.0    0.633054
    2.0    0.167627
    3.0    0.074123
    4.0    0.018072
    5.0    0.006286
    6.0    0.001572
    7.0    0.000262
    Name: id, dtype: float64




```python
(property_df.groupby(['bedrooms']).id.count()/property_df.shape[0]).plot(kind="bar")
```




    <AxesSubplot:xlabel='bedrooms'>




    
![png](output_179_1.png)
    


63% of listings offer 1 bedroom while 16.7% have 2 bedrooms. 9% have no bedrooms at all


```python
amenities_df=property_df['amenities']

```

Ammenities are in textual form but can be made into categories since there are frequently reccurring features. Also we can use wordcloud to map some of the most frequently occuring words


```python
# drop empty rows in amenities
amenities_df = amenities_df[amenities_df != '{}']
```


```python
# create a list of amenities, removing the {} and "" and split each word with a comma
amenities_list = []

for index, row in amenities_df.items():
    amenities_list.append(row.replace('{','').replace('}','').replace('"','').split(','))

amenities_list[:3]
```




    [['TV',
      'Cable TV',
      'Internet',
      'Wireless Internet',
      'Air Conditioning',
      'Kitchen',
      'Heating',
      'Family/Kid Friendly',
      'Washer',
      'Dryer'],
     ['TV',
      'Internet',
      'Wireless Internet',
      'Kitchen',
      'Free Parking on Premises',
      'Buzzer/Wireless Intercom',
      'Heating',
      'Family/Kid Friendly',
      'Washer',
      'Dryer',
      'Smoke Detector',
      'Carbon Monoxide Detector',
      'First Aid Kit',
      'Safety Card',
      'Fire Extinguisher',
      'Essentials'],
     ['TV',
      'Cable TV',
      'Internet',
      'Wireless Internet',
      'Air Conditioning',
      'Kitchen',
      'Free Parking on Premises',
      'Pets Allowed',
      'Pets live on this property',
      'Dog(s)',
      'Cat(s)',
      'Hot Tub',
      'Indoor Fireplace',
      'Heating',
      'Family/Kid Friendly',
      'Washer',
      'Dryer',
      'Smoke Detector',
      'Carbon Monoxide Detector',
      'Essentials',
      'Shampoo']]



Now that we have ammenities as a list, we can add this as a column and one hot encode it to represent the categorical variables ( Stack overflow and kaggle ) have explanations on how this is done. We will use that snippet from there


```python
new_amenities_df = pd.Series(amenities_list, name = 'amenities').to_frame()
new_amenities_df[:5]
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
      <th>amenities</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[TV, Cable TV, Internet, Wireless Internet, Ai...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[TV, Internet, Wireless Internet, Kitchen, Fre...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[TV, Cable TV, Internet, Wireless Internet, Ai...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[Internet, Wireless Internet, Kitchen, Indoor ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[TV, Cable TV, Internet, Wireless Internet, Ki...</td>
    </tr>
  </tbody>
</table>
</div>




```python
dummies_amenities_df = new_amenities_df.drop('amenities', 1).join(
    pd.get_dummies(
        pd.DataFrame(new_amenities_df.amenities.tolist()).stack()
    ).astype(int).sum(level=0)
)

dummies_amenities_df
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
      <th>24-Hour Check-in</th>
      <th>Air Conditioning</th>
      <th>Breakfast</th>
      <th>Buzzer/Wireless Intercom</th>
      <th>Cable TV</th>
      <th>Carbon Monoxide Detector</th>
      <th>Cat(s)</th>
      <th>Dog(s)</th>
      <th>Doorman</th>
      <th>Dryer</th>
      <th>...</th>
      <th>Safety Card</th>
      <th>Shampoo</th>
      <th>Smoke Detector</th>
      <th>Smoking Allowed</th>
      <th>Suitable for Events</th>
      <th>TV</th>
      <th>Washer</th>
      <th>Washer / Dryer</th>
      <th>Wheelchair Accessible</th>
      <th>Wireless Internet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3768</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3769</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3770</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3771</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3772</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>3773 rows × 41 columns</p>
</div>



Now we have all the buzz words as columns


```python
dummies_amenities_df.columns
```




    Index(['24-Hour Check-in', 'Air Conditioning', 'Breakfast',
           'Buzzer/Wireless Intercom', 'Cable TV', 'Carbon Monoxide Detector',
           'Cat(s)', 'Dog(s)', 'Doorman', 'Dryer', 'Elevator in Building',
           'Essentials', 'Family/Kid Friendly', 'Fire Extinguisher',
           'First Aid Kit', 'Free Parking on Premises', 'Gym', 'Hair Dryer',
           'Hangers', 'Heating', 'Hot Tub', 'Indoor Fireplace', 'Internet', 'Iron',
           'Kitchen', 'Laptop Friendly Workspace', 'Lock on Bedroom Door',
           'Other pet(s)', 'Pets Allowed', 'Pets live on this property', 'Pool',
           'Safety Card', 'Shampoo', 'Smoke Detector', 'Smoking Allowed',
           'Suitable for Events', 'TV', 'Washer', 'Washer / Dryer',
           'Wheelchair Accessible', 'Wireless Internet'],
          dtype='object')




```python
dummies_amenities_df.sum()
```




    24-Hour Check-in               616
    Air Conditioning               677
    Breakfast                      291
    Buzzer/Wireless Intercom       538
    Cable TV                      1446
    Carbon Monoxide Detector      2485
    Cat(s)                         382
    Dog(s)                         509
    Doorman                         85
    Dryer                         2997
    Elevator in Building           785
    Essentials                    3237
    Family/Kid Friendly           1963
    Fire Extinguisher             2196
    First Aid Kit                 1680
    Free Parking on Premises      2167
    Gym                            442
    Hair Dryer                     774
    Hangers                        846
    Heating                       3627
    Hot Tub                        303
    Indoor Fireplace               886
    Internet                      2811
    Iron                           742
    Kitchen                       3423
    Laptop Friendly Workspace      745
    Lock on Bedroom Door           100
    Other pet(s)                    51
    Pets Allowed                   472
    Pets live on this property     883
    Pool                           159
    Safety Card                    727
    Shampoo                       2670
    Smoke Detector                3281
    Smoking Allowed                 82
    Suitable for Events            209
    TV                            2574
    Washer                        2992
    Washer / Dryer                   2
    Wheelchair Accessible          300
    Wireless Internet             3667
    dtype: int64




```python
dummies_amenities_df.sum().sort_values(ascending = False).plot(kind='bar', figsize = (15,5));
```


    
![png](output_191_0.png)
    


There is a long list of amenities on offer, washer/Dryer has only  occurences, it might be the only amenity that is insignificant


```python
amenitiesDF = listings_df[['amenities','price','id',]]
amenitiesDFTopper = amenitiesDF.sort_values('price',ascending=[0])
amenitiesDFtop=amenitiesDFTopper.head(30)
allemenities = ''
for index,row in amenitiesDFtop.iterrows():
    p = re.sub('[^a-zA-Z]+',' ', row['amenities'])
    allemenities+=p

allemenities_data=nltk.word_tokenize(allemenities)
filtered_data=[word for word in allemenities_data if word not in stopwords.words('english')] 
wnl = nltk.WordNetLemmatizer() 
allemenities_data=[wnl.lemmatize(data) for data in filtered_data]
allemenities_words=' '.join(allemenities_data)
```


```python
plot_display(allemenities_words)
```


    
![png](output_194_0.png)
    


Of all the amenities that we found and listed, the ones shown on the wordmap are the ones that are strongly related to price. Some of the main ones include Wireless Internet, smoke detector, Monoxide detector, Carbon monoxide detector, Essentials like shampoo, Free Parking and Family and kid friendly. These improve customer experience and also contribute to pricing. The more ammenities the host adds the pricier the listing

## EDA

While we have already done most of the EDA during data understanding, there's a few aspects that we haven't done. For instance, on the calender data set, we have not investigated the availability and price in relation to day of the week, week of the year, month of the year and year. We also have yet to determine if there are factors/ features that relate to or affect price. We also want to uncover neighbourhood trends, find out which times are busiest in Seattle and how the price is affected by those changes. In answer to some of our data mining objectives


```python
fig = plt.figure(figsize = (15,10))
ax = fig.gca()
listings_df.plot.scatter('longitude', 'latitude', ax=ax)
```




    <AxesSubplot:xlabel='longitude', ylabel='latitude'>




    
![png](output_198_1.png)
    


#### Revisiting the review Data


```python
nltk.download('vader_lexicon')

!pip install langdetect
```

    [nltk_data] Downloading package vader_lexicon to
    [nltk_data]     /home/fucking/nltk_data...
    [nltk_data]   Package vader_lexicon is already up-to-date!


    Requirement already satisfied: langdetect in /home/fucking/anaconda/lib/python3.8/site-packages (1.0.9)
    Requirement already satisfied: six in /home/fucking/anaconda/lib/python3.8/site-packages (from langdetect) (1.15.0)



```python
reviewsDF=reviews_df
```

We want to understand the general sentiments in the comments whether they were positive, negative or neutral. In order to do that, we use a built-in NLTK library. To read and score each comment on a polariser



```python
# Our dataframe consists of reviews in different language as well.Hence removing the comments which are not in english
from langdetect import detect

def detect_lang(sente):
    sente=str(sente)
    try:
        return detect(sente)
    except:
        return "None"

for index,row in reviewsDF.iterrows():
    lang=detect_lang(row['comments'])
    reviewsDF.at[index,'language'] = lang
#     print(lang)
    
#taking rows whose language is English
EngReviewsDF=reviewsDF[reviewsDF.language=='en']

EngReviewsDF.head[:5]
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-73-eb087a4ca30d> in <module>
         17 EngReviewsDF=reviewsDF[reviewsDF.language=='en']
         18 
    ---> 19 EngReviewsDF.head[:5]
    

    TypeError: 'method' object is not subscriptable


We check for positive polarity in the sentiments  on the reviews section


```python
polarDF=EngReviewsDF[['pos']]
polarDF=polarDF.groupby(pd.cut(polarDF["pos"], np.arange(0, 1.1, 0.1))).count()
polarDF=polarDF.rename(columns={'pos':'count_of_Comments'})
polarDF=polarDF.reset_index()
polarDF=polarDF.rename(columns={'pos':'range_i'})
for i,r in polarDF.iterrows():
    polarDF.at[i,'RANGE'] = float(str(r['range_i'])[1:4].replace(',',''))
    polarDF.at[i,'Sentiment'] = 'positive'
del polarDF['range_i']
polarDF.head()
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
      <th>count_of_Comments</th>
      <th>RANGE</th>
      <th>Sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>202</td>
      <td>0.0</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1792</td>
      <td>0.1</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3712</td>
      <td>0.2</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2890</td>
      <td>0.3</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1319</td>
      <td>0.4</td>
      <td>positive</td>
    </tr>
  </tbody>
</table>
</div>



We check for negative polarity as well


```python
polarDFneg=EngReviewsDF[['neg']]
polarDFneg=polarDFneg.groupby(pd.cut(polarDFneg["neg"], np.arange(0, 1.1, 0.1))).count()
polarDFneg=polarDFneg.rename(columns={'neg':'count_of_Comments'})
polarDFneg=polarDFneg.reset_index()
polarDFneg=polarDFneg.rename(columns={'neg':'range_i'})
for i,r in polarDFneg.iterrows():
    polarDFneg.at[i,'RANGE'] = float(str(r['range_i'])[1:4].replace(',',''))
    polarDFneg.at[i,'Sentiment'] = 'negative'
del polarDFneg['range_i']
for i,r in polarDFneg.iterrows():
    polarDF=polarDF.append(pd.Series([r[0],r[1],r[2]],index=['count_of_Comments','RANGE','Sentiment']),ignore_index=True)
    
polarDFneg.head()
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
      <th>count_of_Comments</th>
      <th>RANGE</th>
      <th>Sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3180</td>
      <td>0.0</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>1</th>
      <td>130</td>
      <td>0.1</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>0.2</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0.3</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0.4</td>
      <td>negative</td>
    </tr>
  </tbody>
</table>
</div>



Checking for neutral sentiments as well


```python
polarDFneut=EngReviewsDF[['neu']]
polarDFneut=polarDFneut.groupby(pd.cut(polarDFneut["neu"], np.arange(0, 1.0, 0.1))).count()
polarDFneut=polarDFneut.rename(columns={'neu':'count_of_Comments'})
polarDFneut=polarDFneut.reset_index()
polarDFneut=polarDFneut.rename(columns={'neu':'range_i'})
for i,r in polarDFneut.iterrows():
    polarDFneut.at[i,'RANGE'] = float(str(r['range_i'])[1:4].replace(',',''))
    polarDFneut.at[i,'Sentiment'] = 'neutral' 
del polarDFneut['range_i']

for i,r in polarDFneut.iterrows():
    polarDF=polarDF.append(pd.Series([r[0],r[1],r[2]],index=['count_of_Comments','RANGE','Sentiment']),ignore_index=True)
    
polarDFneut.head()

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
      <th>count_of_Comments</th>
      <th>RANGE</th>
      <th>Sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.0</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>1</th>
      <td>17</td>
      <td>0.1</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>2</th>
      <td>83</td>
      <td>0.2</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>3</th>
      <td>208</td>
      <td>0.3</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>4</th>
      <td>560</td>
      <td>0.4</td>
      <td>neutral</td>
    </tr>
  </tbody>
</table>
</div>



Then we plot the polarity of the sentiments against the number of compliments to see the general sentiment


```python
plt.figure(figsize=(10,10))
sns.factorplot(data=polarDF, x="RANGE", y="count_of_Comments",col="Sentiment") 
```

    /home/fucking/anaconda/lib/python3.8/site-packages/seaborn/categorical.py:3704: UserWarning: The `factorplot` function has been renamed to `catplot`. The original name will be removed in a future release. Please update your code. Note that the default `kind` in `factorplot` (`'point'`) has changed `'strip'` in `catplot`.
      warnings.warn(msg)





    <seaborn.axisgrid.FacetGrid at 0x7f5da14bc3d0>




    <Figure size 720x720 with 0 Axes>



    
![png](output_211_3.png)
    


There are very few negative comments, neutral sentiments are skewed towards the range. 


```python
inputDF = pd.read_csv("listings.csv", delimiter=',')

inputDF=inputDF[['number_of_reviews','price','review_scores_rating']]

# replacing NaN values with 0
inputDF.fillna(0, inplace=True)

#Extracting prices from the table
price = inputDF['price']
prices=[]

#clean the data to make it float
for p in price:
    p=float(p[1:].replace(',',''))
    prices.append(p)

#replace the price column with the new column
inputDF['price']=prices

price_review = inputDF[['number_of_reviews', 'price']].sort_values(by = 'price')

price_review.plot(x = 'price', 
                  y = 'number_of_reviews', 
                  style = 'o',
                  figsize =(12,8),
                  legend = False,
                  title = 'Reviews based on Price')

plt.xlabel("price")
plt.ylabel("Number of reviews")
```




    Text(0, 0.5, 'Number of reviews')




    
![png](output_213_1.png)
    


There are more reviews concentrated around a certain price range of between up to $ 250. We see here that customers prefer cheaper listings and use cheaper listings more hence the many reviews

#### Revisiting the calender data


```python
calender_df[:5]
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
      <th>listing_id</th>
      <th>date</th>
      <th>available</th>
      <th>price</th>
      <th>dayofweek</th>
      <th>weekofyear</th>
      <th>month</th>
      <th>year</th>
      <th>day_Name</th>
      <th>holiday</th>
      <th>us_holidays_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>241032</td>
      <td>2016-01-04</td>
      <td>1</td>
      <td>85.000000</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2016</td>
      <td>default</td>
      <td>False</td>
      <td>working</td>
    </tr>
    <tr>
      <th>1</th>
      <td>241032</td>
      <td>2016-01-05</td>
      <td>1</td>
      <td>85.000000</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2016</td>
      <td>default</td>
      <td>False</td>
      <td>working</td>
    </tr>
    <tr>
      <th>2</th>
      <td>241032</td>
      <td>2016-01-06</td>
      <td>0</td>
      <td>137.090652</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2016</td>
      <td>default</td>
      <td>False</td>
      <td>working</td>
    </tr>
    <tr>
      <th>3</th>
      <td>241032</td>
      <td>2016-01-07</td>
      <td>0</td>
      <td>137.090652</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2016</td>
      <td>default</td>
      <td>False</td>
      <td>working</td>
    </tr>
    <tr>
      <th>4</th>
      <td>241032</td>
      <td>2016-01-08</td>
      <td>0</td>
      <td>137.090652</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>2016</td>
      <td>default</td>
      <td>False</td>
      <td>working</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, axs = plt.subplots(1, 3, figsize=(16, 5))
sns.boxplot(data=calender_df, x='month', y='price', ax=axs[0], fliersize=0)
sns.boxplot(data=calender_df, x='dayofweek', y='price', ax=axs[1], fliersize=0)
sns.boxplot(data=calender_df, x='year', y='price', ax=axs[2], fliersize=0)
for ax in axs:
    ax.set_ylim(0, 350)
```


    
![png](output_217_0.png)
    


The price of listings is highest between day 3 and 4 of the week, and month 6 to 9 of the year.


```python
# we modify the data to include average price and a year and month column
yearDF=calender_df.groupby(['year','month']).price.mean()
yearDF=yearDF.reset_index()
yearDF=yearDF.rename(columns={'price':'average_Price'})
yearDF['year-Month']=yearDF['year'].map(str) + "-" + yearDF['month'].map(str)
yearDF.to_csv('./year_month_data.csv')
yearDF[::5]
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
      <th>year</th>
      <th>month</th>
      <th>average_Price</th>
      <th>year-Month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016</td>
      <td>1</td>
      <td>128.293029</td>
      <td>2016-1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2016</td>
      <td>6</td>
      <td>143.543198</td>
      <td>2016-6</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2016</td>
      <td>11</td>
      <td>135.396148</td>
      <td>2016-11</td>
    </tr>
  </tbody>
</table>
</div>




```python
yearDF.plot(kind="bar",x='year-Month', y='average_Price',color = '#662dff', figsize =(15,8), 
           title = 'Seattle Airbnb prices trend over months')
plt.ylabel('Average Price')
```




    Text(0, 0.5, 'Average Price')




    
![png](output_220_1.png)
    


The average price is more or less the same throughout 2016 with a hike between June and Sept as seen earlier in the analysis


```python
!pip install holidays
```

    Requirement already satisfied: holidays in /home/fucking/anaconda/lib/python3.8/site-packages (0.11.1)
    Requirement already satisfied: hijri-converter in /home/fucking/anaconda/lib/python3.8/site-packages (from holidays) (2.1.1)
    Requirement already satisfied: korean-lunar-calendar in /home/fucking/anaconda/lib/python3.8/site-packages (from holidays) (0.2.1)
    Requirement already satisfied: convertdate>=2.3.0 in /home/fucking/anaconda/lib/python3.8/site-packages (from holidays) (2.3.2)
    Requirement already satisfied: six in /home/fucking/anaconda/lib/python3.8/site-packages (from holidays) (1.15.0)
    Requirement already satisfied: python-dateutil in /home/fucking/anaconda/lib/python3.8/site-packages (from holidays) (2.8.1)
    Requirement already satisfied: pymeeus<=1,>=0.3.13 in /home/fucking/anaconda/lib/python3.8/site-packages (from convertdate>=2.3.0->holidays) (0.5.11)
    Requirement already satisfied: pytz>=2014.10 in /home/fucking/anaconda/lib/python3.8/site-packages (from convertdate>=2.3.0->holidays) (2020.1)


Another useful library for calender analysis is to install the holiday library so that we know each holiday in the US to see if the days and or dates with higher prices corresponded to holidays


```python
# We make a copy of the dataset so that we mess with the copy and preserve the original
calendarDF=calender_df.copy()
```


```python
#replacing NaN values with 0
calendarDF.fillna(0, inplace=True)
calendarDF = calendarDF[calendarDF.price != 0]

#Extracting prices from the table
price = calendarDF['price']
prices=[]

for p in price:
    p = re.sub('[^0-9.]+','', p)
    prices.append(float(p))
    
#replace the price column with the new column
calendarDF['price']=prices

calendarDF = calendarDF[calendarDF.price >= 0]

#separating date column into day month and year
calendarDF['Year'],calendarDF['Month'],calendarDF['Day']=calendarDF['date'].str.split('-',2).str
calendarDF.head()
```

    <ipython-input-629-91ebf8a550a9>:19: FutureWarning: Columnar iteration over characters will be deprecated in future releases.
      calendarDF['Year'],calendarDF['Month'],calendarDF['Day']=calendarDF['date'].str.split('-',2).str





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
      <th>listing_id</th>
      <th>date</th>
      <th>available</th>
      <th>price</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>241032</td>
      <td>2016-01-04</td>
      <td>t</td>
      <td>85.0</td>
      <td>2016</td>
      <td>01</td>
      <td>04</td>
    </tr>
    <tr>
      <th>1</th>
      <td>241032</td>
      <td>2016-01-05</td>
      <td>t</td>
      <td>85.0</td>
      <td>2016</td>
      <td>01</td>
      <td>05</td>
    </tr>
    <tr>
      <th>9</th>
      <td>241032</td>
      <td>2016-01-13</td>
      <td>t</td>
      <td>85.0</td>
      <td>2016</td>
      <td>01</td>
      <td>13</td>
    </tr>
    <tr>
      <th>10</th>
      <td>241032</td>
      <td>2016-01-14</td>
      <td>t</td>
      <td>85.0</td>
      <td>2016</td>
      <td>01</td>
      <td>14</td>
    </tr>
    <tr>
      <th>14</th>
      <td>241032</td>
      <td>2016-01-18</td>
      <td>t</td>
      <td>85.0</td>
      <td>2016</td>
      <td>01</td>
      <td>18</td>
    </tr>
  </tbody>
</table>
</div>



Now that we have recreated the calender data set like we did before and seperated the datetime to day, month and year, we can use these for our EDA. Having installed the holidays, we add them to the data set


```python
from datetime import date
import datetime
import calendar
import holidays

calendarDF.fillna(0, inplace=True)
us_holidays = holidays.UnitedStates()

calendarDF['day_Name']='default'
calendarDF['holiday']='False'
calendarDF['us_holidays_name']='working'
for index,row in calendarDF.iterrows():
    sdate = datetime.date(int(row['Year']),int(row['Month']),int(row['Day']))
    vall=date(int(row['Year']),int(row['Month']),int(row['Day'])) in us_holidays
    calendarDF.at[index,'day_Name'] = calendar.day_name[sdate.weekday()]
    calendarDF.at[index,'holiday'] = vall
    calendarDF.at[index,'us_holidays_name'] = us_holidays.get(sdate)
calendarDF.to_csv('./holidays_data.csv')
calendarDF.head()
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
      <th>listing_id</th>
      <th>date</th>
      <th>available</th>
      <th>price</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
      <th>day_Name</th>
      <th>holiday</th>
      <th>us_holidays_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>241032</td>
      <td>2016-01-04</td>
      <td>t</td>
      <td>85.0</td>
      <td>2016</td>
      <td>01</td>
      <td>04</td>
      <td>Monday</td>
      <td>False</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>241032</td>
      <td>2016-01-05</td>
      <td>t</td>
      <td>85.0</td>
      <td>2016</td>
      <td>01</td>
      <td>05</td>
      <td>Tuesday</td>
      <td>False</td>
      <td>None</td>
    </tr>
    <tr>
      <th>9</th>
      <td>241032</td>
      <td>2016-01-13</td>
      <td>t</td>
      <td>85.0</td>
      <td>2016</td>
      <td>01</td>
      <td>13</td>
      <td>Wednesday</td>
      <td>False</td>
      <td>None</td>
    </tr>
    <tr>
      <th>10</th>
      <td>241032</td>
      <td>2016-01-14</td>
      <td>t</td>
      <td>85.0</td>
      <td>2016</td>
      <td>01</td>
      <td>14</td>
      <td>Thursday</td>
      <td>False</td>
      <td>None</td>
    </tr>
    <tr>
      <th>14</th>
      <td>241032</td>
      <td>2016-01-18</td>
      <td>t</td>
      <td>85.0</td>
      <td>2016</td>
      <td>01</td>
      <td>18</td>
      <td>Monday</td>
      <td>True</td>
      <td>Martin Luther King Jr. Day</td>
    </tr>
  </tbody>
</table>
</div>



We estimate and calculate the average price of listings per day to see the trend of prices per day of the week


```python
#calculating Average price for each day

dayDF=calendarDF.groupby('day_Name').price.mean()
dayDF=dayDF.reset_index()
dayDF['day_num']=0

for index,row in dayDF.iterrows():
    if row['day_Name']=='Monday':
        dayDF.at[index,'day_num']=1
    if row['day_Name']=='Tuesday':
        dayDF.at[index,'day_num']=2
    if row['day_Name']=='Wednesday':
        dayDF.at[index,'day_num']=3
    if row['day_Name']=='Thursday':
        dayDF.at[index,'day_num']=4
    if row['day_Name']=='Friday':
        dayDF.at[index,'day_num']=5
    if row['day_Name']=='Saturday':
        dayDF.at[index,'day_num']=6
    if row['day_Name']=='Sunday':
        dayDF.at[index,'day_num']=7
dayDF=dayDF.sort_values('day_num',ascending=[1])
dayDF=dayDF.rename(columns={'price':'Average_Price'})
dayDF
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
      <th>day_Name</th>
      <th>Average_Price</th>
      <th>day_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Monday</td>
      <td>135.676414</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Tuesday</td>
      <td>135.408764</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Wednesday</td>
      <td>135.447880</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Thursday</td>
      <td>136.476032</td>
      <td>4</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Friday</td>
      <td>143.036294</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Saturday</td>
      <td>143.202136</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sunday</td>
      <td>136.459941</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>



What we see is that the average price is higher on weekends between Friday and Saturday


```python
import matplotlib.pyplot as plt
dayname = list()
for i in dayDF['day_Name']:
    dayname.append(i)
avgprice = list()
for i in dayDF['Average_Price']:
    avgprice.append(i)
graph_input = dict(zip(dayname,avgprice))

plt.scatter(dayname,avgprice)
plt.show()
```


    
![png](output_231_0.png)
    



```python
calendarDF.groupby('us_holidays_name').listing_id.count()
```




    us_holidays_name
    Christmas Day                 2829
    Christmas Day (Observed)      2831
    Columbus Day                  2650
    Independence Day              2371
    Labor Day                     2544
    Martin Luther King Jr. Day    2231
    Memorial Day                  2583
    New Year's Day                2922
    New Year's Day (Observed)     2776
    Thanksgiving                  2746
    Veterans Day                  2718
    Washington's Birthday         2529
    Name: listing_id, dtype: int64



In general, all holidays have bookings with New Year's day being the highest.


```python
calendarDF.groupby('us_holidays_name').price.mean()
```




    us_holidays_name
    Christmas Day                 137.974903
    Christmas Day (Observed)      137.895797
    Columbus Day                  134.985660
    Independence Day              151.402362
    Labor Day                     142.087264
    Martin Luther King Jr. Day    121.740475
    Memorial Day                  143.233837
    New Year's Day                138.126968
    New Year's Day (Observed)     135.572767
    Thanksgiving                  136.054989
    Veterans Day                  140.070272
    Washington's Birthday         125.353895
    Name: price, dtype: float64



Independence day has the highest price, Martin Luther King Jr Day and Washington's Birthday being the cheapest.

Independence day affectionately known as the 4th of July in the USA is sometimes celebrated as a long weekend. So we check the average prices over the weekends. Especially around July as trends have shown a spike in prices between June and August.


```python
# analyzing data from date 4th of July to date 13th of July which includes both long weekend 
#and normal workdays to compare prices 


marDF=calendarDF[(calendarDF['Year'] == '2016') & (calendarDF['Month'] == '07' )& 
                 ((calendarDF['Day'] == '04' )|(calendarDF['Day'] == '05' )|(calendarDF['Day'] == '06' )
                  | (calendarDF['Day'] == '07' )| (calendarDF['Day'] == '08' )| (calendarDF['Day'] == '09' )
                  | (calendarDF['Day'] == '10' )| (calendarDF['Day'] == '11' )| (calendarDF['Day'] == '12' )
                  | (calendarDF['Day'] == '13' ))]
marDF=marDF.groupby('Day').price.mean()
marDF=marDF.reset_index()
marDF=marDF.sort_values('Day',ascending=[1])
marDF=marDF.rename(columns={'price':'Average_Price'})
marDF.head(10)
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
      <th>Day</th>
      <th>Average_Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>04</td>
      <td>151.402362</td>
    </tr>
    <tr>
      <th>1</th>
      <td>05</td>
      <td>150.522832</td>
    </tr>
    <tr>
      <th>2</th>
      <td>06</td>
      <td>150.200837</td>
    </tr>
    <tr>
      <th>3</th>
      <td>07</td>
      <td>150.709570</td>
    </tr>
    <tr>
      <th>4</th>
      <td>08</td>
      <td>156.260469</td>
    </tr>
    <tr>
      <th>5</th>
      <td>09</td>
      <td>156.448161</td>
    </tr>
    <tr>
      <th>6</th>
      <td>10</td>
      <td>150.302538</td>
    </tr>
    <tr>
      <th>7</th>
      <td>11</td>
      <td>149.866250</td>
    </tr>
    <tr>
      <th>8</th>
      <td>12</td>
      <td>150.058504</td>
    </tr>
    <tr>
      <th>9</th>
      <td>13</td>
      <td>150.656785</td>
    </tr>
  </tbody>
</table>
</div>



Clearly and conclusively, weekends cost more than week days. A plot of average price per day of the week should show this


```python
x=marDF['Day'].tolist()
y=marDF['Average_Price'].tolist()
plt.plot(x,y, 'ro-')
plt.ylabel('Average Price')
plt.xlabel('Days')
plt.show()
```


    
![png](output_239_0.png)
    


This answers our data mining question of what the **busiest times in Seattle are and how the prices change. From our analysis, the summer is the busiest times in seattle and prices are higher**. The Independence day holiday is the most expensive to visit in though holidays in general are pricier. Good holidays to visit seattle on are Martin Luther King Jr and Washington's birthday. Weekends are also pricer than week days

Since we have NLT we can utilise it to process the text features


```python
summary = listings_df[['summary','price']]
summary = summary[pd.notnull(summary['summary'])]
summary = summary[summary['summary']!=0]
summary = summary.sort_values('price',ascending=[0])
top100DF = summary.head(1000)
```


```python
from nltk.corpus import stopwords
import string

words=''
for index,row in top100DF.iterrows():
    words += row['summary']
    
string_punctuation = string.punctuation
ignoreChar=['\r','\n','',' ',"'s"]
nums=['0','1','2','3','4','5','6','7','8','9']
summary_data=nltk.word_tokenize(words)
words_only = [l.lower() for l in summary_data if l not in string_punctuation if l not in ignoreChar if l not in nums]
filtered_data=[word for word in words_only if word not in stopwords.words('english')] 
wnl = nltk.WordNetLemmatizer() 
final_data=[wnl.lemmatize(data) for data in filtered_data]
final_words=' '.join(final_data)
final_words[:500]
```




    '100 walk score close convention center westlake station pike place market amazon belltown capitol hill spotless reliable accommodating professional host apartment lower level classic 1915 craftsman bungalow heart wallingford walk score 93 ideally located 45th 50th block restaurant grocery store drug stores.walk downtown seattle attraction charming b b tree-lined side street capitol hill neighborhood.awesome two bedroom condo perfect convention traveler business traveler couple wanting see enjoy '




```python
plot_display(final_words)
```


    
![png](output_244_0.png)
    


The most frequently used words in listing summary are home, room, downtown, apartment,neighbourhd, garden, ballard bus line, bathroom. Hosts listing houses should use these key words to get more customers

#### Revisiting Price related metrics


```python
some_features=listings_df[['id','property_type','price','room_type','beds','bedrooms',
                          'neighbourhood_cleansed','bathrooms','accommodates','latitude','longitude']]
some_features.fillna(0,inplace=True)
some_features['price'] = some_features['price'].str.extract(r'(\d+)', expand=False)
# make price float
some_features['price'] = some_features['price'].astype(float, errors = 'raise')
some_features = some_features[some_features.bathrooms >0]
some_features = some_features[some_features.beds >0]
some_features = some_features[some_features.price >0]
some_features = some_features[some_features.accommodates >0]
some_features.head()
```


```python
plt.figure(figsize=(18,22))
ax = sns.heatmap(some_features.groupby([
        'neighbourhood_cleansed', 'bedrooms']).price.mean().unstack(),annot=True, fmt=".0f")
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
```




    (87.5, -0.5)




    
![png](output_248_1.png)
    


The heatmap shows that the number of bed rooms greatly affect price, the more the bedrooms the higher the price but neighbourhood is also important. As expected some neighbourhoods with fewer bedrooms will be more expensive vice versa


```python
# apartments are the most expensive room type,in 
plt.figure(figsize=(12,12))
sns.heatmap(some_features.groupby(['property_type', 'room_type']).price.mean().unstack(),annot=True, fmt=".0f")
```




    <AxesSubplot:xlabel='room_type', ylabel='property_type'>




    
![png](output_250_1.png)
    


The room type certainly affects the price, apartments are the most expensive followed by boats.


```python
some_features.groupby(['bedrooms']).price.mean().plot(kind="bar")
```




    <AxesSubplot:xlabel='bedrooms'>




    
![png](output_252_1.png)
    


### Key Findings from the Analysis of the data

 When we started we sort to understand and answer these questions:
   1. What features of the listings are related to price?    
   2. What vibes are happening in each neighbourhood?    
   3. Are there factors/metrics for that help understand customer experience?    
   4. Are there things that guests/AirBnB can do to boost guest experience and get more clients?
   5. Has there been an increase in the number of AirBnb hosts?
   6. What are the busiest times of the year in seattle and by how much do the prices change
   
Our Analysis of the data has uncovered these findings. 
 
 **Price**
  + The neighbourhood, number of beds, how many nights the guest will be staying, whether they have extra guests, the day of the week, the months, the neighbourhood, whether its a holiday and Amenities provided
  + Of all the amenities that we found and listed, the ones shown on the wordmap are the ones that are strongly related to price. Some of the main ones include *Wireless Internet, smoke detector, Monoxide detector, Carbon monoxide detector, Essentials like shampoo, Free Parking and Family and kid friendly*. These improve customer experience and also contribute to pricing. **The more ammenities the host adds the pricier the listing**
  + Appartments and houses are the most common properties with 45% and 44.7% of all the listings of room types and they are the most expensive
  
**Vibes in each neighbourhood**
  + Broadway is the most frequently listed neighbourhood which occurs 397 times
  + We see that there are zipcodes that have more listings like the 98122, 98103 and 98102. These are generally higher, the bulk of the listings are on average the same until we get to 98104. Then we see 98199, 126,106,108,133,136 have about the same listings around the 50-60 listings average
  
**Has there been an increase in AirBnb's hosts
  + Since 2008 to 2016. In 10 years, there was a growth from 4 listings to 3816 listings, the growth is exponential. Between 2009 and 2011 the growth rate doubled at a average growth rate of 140% per year. From 2011 the growth rate steadied to an average growth rate of 30%. About 50% of hosts joined July 2013
  
**Busiest times in Seattle and how prices change**
  + The summer is the busiest times in seattle, between June and August. Prices are highest during these times. Holidays are generally higher priced on average than normal days.The Independence day holiday is the most expensive to visit in, in Seattle. Good holidays to visit seattle on are Martin Luther King Jr and Washington's birthday, they are cheaper as well. Weekends are also higher priced than week days, prices go up on Friday and Saturday
  
**Customer metrics**
  + reviews per month and review score rating features are useful features for understanding customer experience and thus improving customer experience. As expected some users/hosts have many reviews but fewer reviews per month suggesting they have been listed longer but not neccesarily that they are perfoming better than those with fewer number of ratigs but higher reviews per month

##  Data Preparation

Our business understanding is broad, we sought to understand features that we can link to the customer experience and features that affect pricing. As we come down to preparing the data for modelling, we realise that we need to narrow down to one aspect to be investigated fully first then as time permits we can do the other one. We will start with and narrow our focus on price first. We will model price prediction.

What we have already learned about price from our EDA has revealed the features that seem to be related to or affect pricing. We will focus on these features. We concluded that:

   + Amenities
   + number of reviews
   + propety type
   + room type
   + neighbourhood
   + number of beds
   + how many nights 
   + The guest will be staying
   + The day of the week, the months
   + Whether its a holiday 
   + Amenities provided: and on the ammenities we provided a list of amenities that are essential

We will now clean and prepare the columns with these and related features. Fill in missing values. Convert the categorical variables into one hot encoded dummies

#### Selecting the columns that we want

We have 3 datasets:

    calendar.csv - calendar data for the listings: availability dates, price for each date.
    listings.csv - summary information on listing in Seattle such as: location, host information, cleaning and   guest fees, amenities etc.
    reviews.csv - summary review data for the listings.
We will only use the calender data and listings, the reviews is in textual format so we wont be modelling it. During our EDA we have already predetermined the columns that affect price, so here we just list them


```python
#Select only the columns which we are interested in

selected_listings_cols = [
    'id', 'zipcode', 'property_type', 'room_type', 'accommodates', 'bathrooms', 'bedrooms',
 'beds', 'bed_type', 'amenities', 'price', 'weekly_price', 'monthly_price', 'security_deposit',
 'cleaning_fee', 'number_of_reviews', 'review_scores_rating', 'review_scores_accuracy',
 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication',
 'review_scores_location', 'review_scores_value', 'cancellation_policy',
]

new_listings_df = listings_df[selected_listings_cols]
new_listings_df['price'][:5]

```




    0     85.0
    1    150.0
    2    975.0
    3    100.0
    4    450.0
    Name: price, dtype: float64



**Drop missing values**


```python
# Drop rows with missing values of the number of reviews
# Drop rows with missing values of the bathrooms
# Drop rows with missing values of the property type

review_scores_cols = [col for col in listings_df.columns if 'review_scores' in col]
drop_cols = ['number_of_reviews','bathrooms','property_type'] + review_scores_cols

new_listings_df.dropna(subset = drop_cols, axis = 0, inplace = True)

# Drop rows where the number of reviews is 0.
new_listings_df = new_listings_df[new_listings_df['number_of_reviews'] != 0]

# Drop rows where the amenities are empty
new_listings_df = new_listings_df[new_listings_df['amenities'] != '{}']

# Reset index
new_listings_df = new_listings_df.reset_index(drop = True)
```

    <ipython-input-133-45b57bcb1fc0>:8: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      new_listings_df.dropna(subset = drop_cols, axis = 0, inplace = True)


**Fill missing values**


```python
# Fill missing values of the bedrooms and beds as 1
new_listings_df[['bedrooms', 'beds']] = new_listings_df[['bedrooms', 'beds']].fillna(value = 1)
```


```python
# Change the data type of the price and related columns and fill missing values as 0
new_listings_price_df = new_listings_df[['price','weekly_price','monthly_price','security_deposit','cleaning_fee']]

for col in ['price','weekly_price','monthly_price','security_deposit','cleaning_fee']:
    new_listings_price_df = pd.concat([new_listings_price_df.drop(columns = [col]), new_listings_price_df[col].str.replace('$','').str.replace(',','').astype(float)], axis = 1)
    

```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-135-c8147b92ab2f> in <module>
          3 
          4 for col in ['price','weekly_price','monthly_price','security_deposit','cleaning_fee']:
    ----> 5     new_listings_price_df = pd.concat([new_listings_price_df.drop(columns = [col]), new_listings_price_df[col].str.replace('$','').str.replace(',','').astype(float)], axis = 1)
          6 
          7 new_listings_price_df.fillna(value = 0, inplace =True)


    ~/anaconda/lib/python3.8/site-packages/pandas/core/generic.py in __getattr__(self, name)
       5133             or name in self._accessors
       5134         ):
    -> 5135             return object.__getattribute__(self, name)
       5136         else:
       5137             if self._info_axis._can_hold_identifiers_and_holds_name(name):


    ~/anaconda/lib/python3.8/site-packages/pandas/core/accessor.py in __get__(self, obj, cls)
        185             # we're accessing the attribute of the class, i.e., Dataset.geo
        186             return self._accessor
    --> 187         accessor_obj = self._accessor(obj)
        188         # Replace the property with the accessor object. Inspired by:
        189         # https://www.pydanny.com/cached-property.html


    ~/anaconda/lib/python3.8/site-packages/pandas/core/strings.py in __init__(self, data)
       2098 
       2099     def __init__(self, data):
    -> 2100         self._inferred_dtype = self._validate(data)
       2101         self._is_categorical = is_categorical_dtype(data.dtype)
       2102         self._is_string = data.dtype.name == "string"


    ~/anaconda/lib/python3.8/site-packages/pandas/core/strings.py in _validate(data)
       2155 
       2156         if inferred_dtype not in allowed_types:
    -> 2157             raise AttributeError("Can only use .str accessor with string values!")
       2158         return inferred_dtype
       2159 


    AttributeError: Can only use .str accessor with string values!



```python
new_listings_price_df.fillna(value = 0, inplace =True)
```

    /home/fucking/anaconda/lib/python3.8/site-packages/pandas/core/frame.py:4317: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      return super().fillna(



```python
# Calculate weekly and monthly price
new_listings_price_df['calc_weekly_price'] = new_listings_price_df['price'] * 7
new_listings_price_df['calc_monthly_price'] = new_listings_price_df['price'] * 30
new_listings_price_df[:5]

```

    <ipython-input-158-c05eff8d17ca>:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      new_listings_price_df['calc_weekly_price'] = new_listings_price_df['price'] * 7
    <ipython-input-158-c05eff8d17ca>:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      new_listings_price_df['calc_monthly_price'] = new_listings_price_df['price'] * 30





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
      <th>price</th>
      <th>weekly_price</th>
      <th>monthly_price</th>
      <th>security_deposit</th>
      <th>cleaning_fee</th>
      <th>calc_weekly_price</th>
      <th>calc_monthly_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>85.0</td>
      <td>595.0</td>
      <td>2550.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>595.0</td>
      <td>2550.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>150.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>100.0</td>
      <td>40.0</td>
      <td>1050.0</td>
      <td>4500.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>975.0</td>
      <td>6825.0</td>
      <td>29250.0</td>
      <td>1.0</td>
      <td>300.0</td>
      <td>6825.0</td>
      <td>29250.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>450.0</td>
      <td>3150.0</td>
      <td>13500.0</td>
      <td>700.0</td>
      <td>125.0</td>
      <td>3150.0</td>
      <td>13500.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>120.0</td>
      <td>800.0</td>
      <td>3600.0</td>
      <td>0.0</td>
      <td>40.0</td>
      <td>840.0</td>
      <td>3600.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Fill the weekly and monthky price by its calculated values
for idx, row in new_listings_price_df.iterrows():
    if row['weekly_price'] == 0:
        new_listings_price_df.loc[idx, ['weekly_price']] = row['calc_weekly_price']
    if row['monthly_price'] == 0:
        new_listings_price_df.loc[idx, ['monthly_price']] = row['calc_monthly_price']

new_listings_price_df[:5]
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
      <th>price</th>
      <th>weekly_price</th>
      <th>monthly_price</th>
      <th>security_deposit</th>
      <th>cleaning_fee</th>
      <th>calc_weekly_price</th>
      <th>calc_monthly_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>85.0</td>
      <td>595.0</td>
      <td>2550.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>595.0</td>
      <td>2550.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>150.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>100.0</td>
      <td>40.0</td>
      <td>1050.0</td>
      <td>4500.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>975.0</td>
      <td>6825.0</td>
      <td>29250.0</td>
      <td>1.0</td>
      <td>300.0</td>
      <td>6825.0</td>
      <td>29250.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>450.0</td>
      <td>3150.0</td>
      <td>13500.0</td>
      <td>700.0</td>
      <td>125.0</td>
      <td>3150.0</td>
      <td>13500.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>120.0</td>
      <td>800.0</td>
      <td>3600.0</td>
      <td>0.0</td>
      <td>40.0</td>
      <td>840.0</td>
      <td>3600.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#drop the calculated prices columns
new_listings_price_df.drop(columns = ['calc_weekly_price', 'calc_monthly_price'], inplace = True)
```

    /home/fucking/anaconda/lib/python3.8/site-packages/pandas/core/frame.py:4163: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      return super().drop(



```python
new_listings_price_df
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
      <th>price</th>
      <th>weekly_price</th>
      <th>monthly_price</th>
      <th>security_deposit</th>
      <th>cleaning_fee</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>85.0</td>
      <td>595.0</td>
      <td>2550.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>150.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>100.0</td>
      <td>40.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>975.0</td>
      <td>6825.0</td>
      <td>29250.0</td>
      <td>1.0</td>
      <td>300.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>450.0</td>
      <td>3150.0</td>
      <td>13500.0</td>
      <td>700.0</td>
      <td>125.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>120.0</td>
      <td>800.0</td>
      <td>3600.0</td>
      <td>0.0</td>
      <td>40.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3118</th>
      <td>154.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>150.0</td>
      <td>85.0</td>
    </tr>
    <tr>
      <th>3119</th>
      <td>65.0</td>
      <td>455.0</td>
      <td>1950.0</td>
      <td>0.0</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>3120</th>
      <td>95.0</td>
      <td>600.0</td>
      <td>2.0</td>
      <td>500.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3121</th>
      <td>359.0</td>
      <td>2513.0</td>
      <td>10770.0</td>
      <td>0.0</td>
      <td>230.0</td>
    </tr>
    <tr>
      <th>3122</th>
      <td>79.0</td>
      <td>553.0</td>
      <td>2370.0</td>
      <td>500.0</td>
      <td>50.0</td>
    </tr>
  </tbody>
</table>
<p>3123 rows × 5 columns</p>
</div>



**Categorical values changed to dummy variable**


```python
# Create dummy columns of cancellation policy, room type, property type and bed type

cancellation_policy_dummy_df = pd.get_dummies(new_listings_df['cancellation_policy'], prefix = 'cancellation_policy')
room_type_dummy_df = pd.get_dummies(new_listings_df['room_type'], prefix = 'room_type')
property_type_dummy_df = pd.get_dummies(new_listings_df['property_type'], prefix = 'property_type')
bed_type_dummy_df = pd.get_dummies(new_listings_df['bed_type'], prefix = 'bed_type')
bed_type_dummy_df
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
      <th>bed_type_Airbed</th>
      <th>bed_type_Couch</th>
      <th>bed_type_Futon</th>
      <th>bed_type_Pull-out Sofa</th>
      <th>bed_type_Real Bed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3118</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3119</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3120</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3121</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3122</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>3123 rows × 5 columns</p>
</div>



**Create dummy variables based on amenities**


```python
# Create dummy columns based on the ammenities

# Drop rows with empty rows
amenities_series = new_listings_df['amenities']
amenities_series = amenities_series[amenities_series != '{}']

# Iterate over rows and format them as list
amenities_list = []

for index, row in amenities_series.items():
    amenities_list.append(row.replace('{','').replace('}','').replace('"','').split(','))

# Convert the list to a data frame
amenities_df = pd.Series(amenities_list, name = 'amenities').to_frame()

# Create a dummy data frame
dummies_amenities_df = amenities_df.drop('amenities', 1).join(
    pd.get_dummies(
        pd.DataFrame(amenities_df.amenities.tolist()).stack()
    ).astype(int).sum(level=0)
)

# Reset index
# dummies_amenities_df = dummies_amenities_df.reset_index(drop=True)
dummies_amenities_df[:5]
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
      <th>24-Hour Check-in</th>
      <th>Air Conditioning</th>
      <th>Breakfast</th>
      <th>Buzzer/Wireless Intercom</th>
      <th>Cable TV</th>
      <th>Carbon Monoxide Detector</th>
      <th>Cat(s)</th>
      <th>Dog(s)</th>
      <th>Doorman</th>
      <th>Dryer</th>
      <th>...</th>
      <th>Safety Card</th>
      <th>Shampoo</th>
      <th>Smoke Detector</th>
      <th>Smoking Allowed</th>
      <th>Suitable for Events</th>
      <th>TV</th>
      <th>Washer</th>
      <th>Washer / Dryer</th>
      <th>Wheelchair Accessible</th>
      <th>Wireless Internet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 41 columns</p>
</div>




```python
dummy_df = pd.concat([cancellation_policy_dummy_df, room_type_dummy_df, property_type_dummy_df, bed_type_dummy_df, dummies_amenities_df], axis = 1)
dummy_df[:5]

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
      <th>cancellation_policy_flexible</th>
      <th>cancellation_policy_moderate</th>
      <th>cancellation_policy_strict</th>
      <th>room_type_Entire home/apt</th>
      <th>room_type_Private room</th>
      <th>room_type_Shared room</th>
      <th>property_type_Apartment</th>
      <th>property_type_Bed &amp; Breakfast</th>
      <th>property_type_Boat</th>
      <th>property_type_Bungalow</th>
      <th>...</th>
      <th>Safety Card</th>
      <th>Shampoo</th>
      <th>Smoke Detector</th>
      <th>Smoking Allowed</th>
      <th>Suitable for Events</th>
      <th>TV</th>
      <th>Washer</th>
      <th>Washer / Dryer</th>
      <th>Wheelchair Accessible</th>
      <th>Wireless Internet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 68 columns</p>
</div>



**Concatinate the dummy dataframes together**


```python
concat_listings_df =pd.concat([dummy_df,new_listings_price_df],axis=1)
concat_listings_df ['price']

#pd.concat([new_listings_df.select_dtypes(include = ['int', 'float']), new_listings_price_df, dummy_df], axis = 1)
```




    0        85.0
    1       150.0
    2       975.0
    3       450.0
    4       120.0
            ...  
    3118    154.0
    3119     65.0
    3120     95.0
    3121    359.0
    3122     79.0
    Name: price, Length: 3123, dtype: float64




```python

```


```python
# Create a list which contains always non available listing_id
always_f_listing_id = list(calender_df.groupby('listing_id')['available'].sum().loc[lambda x : x == 0].index.values)
```


```python
calender_df[:5]
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
      <th>listing_id</th>
      <th>date</th>
      <th>available</th>
      <th>price</th>
      <th>dayofweek</th>
      <th>weekofyear</th>
      <th>month</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>241032</td>
      <td>2016-01-04</td>
      <td>1</td>
      <td>85.000000</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>1</th>
      <td>241032</td>
      <td>2016-01-05</td>
      <td>1</td>
      <td>85.000000</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>2</th>
      <td>241032</td>
      <td>2016-01-06</td>
      <td>0</td>
      <td>137.090652</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>3</th>
      <td>241032</td>
      <td>2016-01-07</td>
      <td>0</td>
      <td>137.090652</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>4</th>
      <td>241032</td>
      <td>2016-01-08</td>
      <td>0</td>
      <td>137.090652</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>2016</td>
    </tr>
  </tbody>
</table>
</div>




```python
calender_df.isna().sum()
```




    listing_id    0
    date          0
    available     0
    price         0
    dayofweek     0
    weekofyear    0
    month         0
    year          0
    dtype: int64



## Data Modelling

We will use 
   + concat_listings_df and the calender_df


```python
concat_listings_df.columns
```




    Index(['cancellation_policy_flexible', 'cancellation_policy_moderate',
           'cancellation_policy_strict', 'room_type_Entire home/apt',
           'room_type_Private room', 'room_type_Shared room',
           'property_type_Apartment', 'property_type_Bed & Breakfast',
           'property_type_Boat', 'property_type_Bungalow', 'property_type_Cabin',
           'property_type_Camper/RV', 'property_type_Chalet',
           'property_type_Condominium', 'property_type_Dorm',
           'property_type_House', 'property_type_Loft', 'property_type_Other',
           'property_type_Tent', 'property_type_Townhouse',
           'property_type_Treehouse', 'property_type_Yurt', 'bed_type_Airbed',
           'bed_type_Couch', 'bed_type_Futon', 'bed_type_Pull-out Sofa',
           'bed_type_Real Bed', '24-Hour Check-in', 'Air Conditioning',
           'Breakfast', 'Buzzer/Wireless Intercom', 'Cable TV',
           'Carbon Monoxide Detector', 'Cat(s)', 'Dog(s)', 'Doorman', 'Dryer',
           'Elevator in Building', 'Essentials', 'Family/Kid Friendly',
           'Fire Extinguisher', 'First Aid Kit', 'Free Parking on Premises', 'Gym',
           'Hair Dryer', 'Hangers', 'Heating', 'Hot Tub', 'Indoor Fireplace',
           'Internet', 'Iron', 'Kitchen', 'Laptop Friendly Workspace',
           'Lock on Bedroom Door', 'Other pet(s)', 'Pets Allowed',
           'Pets live on this property', 'Pool', 'Safety Card', 'Shampoo',
           'Smoke Detector', 'Smoking Allowed', 'Suitable for Events', 'TV',
           'Washer', 'Washer / Dryer', 'Wheelchair Accessible',
           'Wireless Internet', 'price', 'weekly_price', 'monthly_price',
           'security_deposit', 'cleaning_fee'],
          dtype='object')




```python

selected_cols = [
 'accommodates',
 'bathrooms',
 'bedrooms',
 'beds',
 'review_scores_accuracy',
 'review_scores_cleanliness',
 'review_scores_checkin',
 'review_scores_communication',
 'review_scores_location',
 'cancellation_policy_flexible',
 'cancellation_policy_moderate',
 'cancellation_policy_strict',
 'room_type_Entire home/apt',
 'room_type_Private room',
 'room_type_Shared room',
 'property_type_Apartment',
 'property_type_Bed & Breakfast',
 'property_type_Boat',
 'property_type_Bungalow',
 'property_type_Cabin',
 'property_type_Camper/RV',
 'property_type_Chalet',
 'property_type_Condominium',
 'property_type_Dorm',
 'property_type_House',
 'property_type_Loft',
 'property_type_Other',
 'property_type_Tent',
 'property_type_Townhouse',
 'property_type_Treehouse',
 'property_type_Yurt',
 'bed_type_Airbed',
 'bed_type_Couch',
 'bed_type_Futon',
 'bed_type_Pull-out Sofa',
 'bed_type_Real Bed',
 '24-Hour Check-in',
 'Air Conditioning',
 'Breakfast',
 'Buzzer/Wireless Intercom',
 'Cable TV',
 'Carbon Monoxide Detector',
 'Cat(s)',
 'Dog(s)',
 'Doorman',
 'Dryer',
 'Elevator in Building',
 'Essentials',
 'Family/Kid Friendly',
 'Fire Extinguisher',
 'First Aid Kit',
 'Free Parking on Premises',
 'Gym',
 'Hair Dryer',
 'Hangers',
 'Heating',
 'Hot Tub',
 'Indoor Fireplace',
 'Internet',
 'Iron',
 'Kitchen',
 'Laptop Friendly Workspace',
 'Lock on Bedroom Door',
 'Other pet(s)',
 'Pets Allowed',
 'Pets live on this property',
 'Pool',
 'Safety Card',
 'Shampoo',
 'Smoke Detector',
 'Smoking Allowed',
 'Suitable for Events',
 'TV',
 'Washer',
 'Wheelchair Accessible',
 'Wireless Internet'
                ]
```

### Machine Learning: Model Selection

We will start with XGBoost first because it is a complex and advanced gradient boosting algorith that can deal with many features ( our data set has too many variables)


```python
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import plot_importance
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
```


```python
X=concat_listings_df.drop(['price'],axis=1).select_dtypes(exclude=['object'])
y=concat_listings_df['price']
```


```python
# feature extraction
test = SelectKBest(score_func=f_classif, k=4)
fit = test.fit(X, y)
# summarize scores
#set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:5,:])
```

    [ 1.53466214  1.2987518   2.90972465 13.47652381  9.74081468  7.69373935
      2.41685525  0.8211285  11.37626548  0.40519549  0.57030881  0.34520482
      0.31617885  0.9355316   1.59113654  2.53268221  0.78594757  1.47508666
      0.39350675  0.95576457  0.97043513  0.09909161  1.74685022  0.96428115
      0.74663051  0.70456078  1.47429514  1.53097578  1.80095279  1.45991194
      1.55088211  2.31376894  1.11652724  1.07495732  0.98874016  2.92842166
      1.17212899  2.27682701  1.02246066  2.30845121  0.93481025  1.24830463
      1.31057935  2.64698341  1.51811673  1.46783559  1.14992775  2.44973911
      1.88447824  1.07653317  1.58715301  1.20123227  1.25092065  1.04704867
      0.44190869  1.13826204  1.55845695  3.05327576  1.22631481  1.24680062
      0.83378803  0.86773784  1.69089912  2.00411682  1.2468099   0.11179384
      1.24166705  0.95109599 11.99159758 13.29630536  2.89720016 11.95987119]
    [[1.000e+00 5.950e+02 2.550e+03 0.000e+00]
     [1.000e+00 1.000e+00 3.000e+00 4.000e+01]
     [1.000e+00 6.825e+03 2.925e+04 3.000e+02]
     [1.000e+00 3.150e+03 1.350e+04 1.250e+02]
     [0.000e+00 8.000e+02 3.600e+03 4.000e+01]]



```python
model = LogisticRegression(solver='lbfgs')
rfe = RFE(model, 3)
fit = rfe.fit(X, y)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)
```

    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/utils/validation.py:67: FutureWarning: Pass n_features_to_select=3 as keyword args. From version 0.25 passing these as positional arguments will result in an error
      warnings.warn("Pass {} as keyword args. From version 0.25 "
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /home/fucking/anaconda/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(



```python
# Train, test and validation data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=99)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=99) 
```


```python
# Create a model and fit the data to it 

xgb_model = XGBRegressor(
    max_depth=15,
    n_estimators=1000,
    min_child_weight=10, 
    colsample_bytree=0.6, 
    subsample=0.6, 
    eta=0.2,    
    seed=0,
    learning_rate = 0.05,
    n_jobs=2)

xgb_model.fit(
    X_train, 
    y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, y_train), (X_val, y_val)], 
    verbose=10, 
    early_stopping_rounds = 5)
```

    [0]	validation_0-rmse:147.14436	validation_1-rmse:146.99689
    [10]	validation_0-rmse:95.54114	validation_1-rmse:95.90320
    [20]	validation_0-rmse:64.75782	validation_1-rmse:66.27467
    [30]	validation_0-rmse:47.27364	validation_1-rmse:48.85769
    [40]	validation_0-rmse:38.16735	validation_1-rmse:41.10921
    [50]	validation_0-rmse:32.86573	validation_1-rmse:36.87308
    [60]	validation_0-rmse:29.69966	validation_1-rmse:34.57921
    [70]	validation_0-rmse:27.70108	validation_1-rmse:33.17731
    [80]	validation_0-rmse:26.40963	validation_1-rmse:32.58391
    [90]	validation_0-rmse:25.15677	validation_1-rmse:31.58672
    [100]	validation_0-rmse:24.32839	validation_1-rmse:31.16549
    [103]	validation_0-rmse:24.15864	validation_1-rmse:31.21159





    XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                 colsample_bynode=1, colsample_bytree=0.6, eta=0.2, gamma=0,
                 gpu_id=-1, importance_type='gain', interaction_constraints='',
                 learning_rate=0.05, max_delta_step=0, max_depth=15,
                 min_child_weight=10, missing=nan, monotone_constraints='()',
                 n_estimators=1000, n_jobs=2, num_parallel_tree=1, random_state=0,
                 reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=0,
                 subsample=0.6, tree_method='exact', validate_parameters=1,
                 verbosity=None)




```python
xgb_train_pred = xgb_model.predict(X_train)
xgb_val_pred = xgb_model.predict(X_val)
xgb_test_pred = xgb_model.predict(X_test)
```


```python
for i in [[y_train, xgb_train_pred], [y_val, xgb_val_pred], [y_test, xgb_test_pred]]:
    print(r2_score(i[0], i[1]))
```

    0.9248414694882681
    0.8683763163400222
    0.916730205761942


The model actually performs relatively well with between 85 to 90% accuracy on the training data and testing data


```python
fig, ax = plt.subplots(figsize=(10, 15))
plot_importance(xgb_model,ax=ax);
```


    
![png](output_296_0.png)
    


The feature importances are not entirely consistent with our EDA findings possibly due to dimensionality problems and too much correlation. Perhaps we could revisit our feature engineering and reduce the features

## Customer Experience prediction

Our other busines objective was to determine if there are metrics that we can use to determine or understand customer experience. We have already done the EDA and feature engineering. We just have to do the final feature selection and model


```python
df=listings_df.copy()

df['cx_score'] = df['review_scores_rating'] / 100 * df['reviews_per_month']

```

We found that there's a strong relation between reviews scores rating and reviews per month. Since we can only have one response variable, we can combine these two by multiply them. We call the column cx_score


```python
listings_df['neighbourhood_cleansed'].value_counts()[:10]
```




    Broadway                     397
    Belltown                     234
    Wallingford                  167
    Fremont                      158
    Minor                        135
    University District          122
    Stevens                      119
    First Hill                   108
    Central Business District    103
    Lower Queen Anne              94
    Name: neighbourhood_cleansed, dtype: int64



We list the top 10 neighbourhoods which we will use to model our data. We will choose one neighbourhood which we will use to model the data with


```python

```
