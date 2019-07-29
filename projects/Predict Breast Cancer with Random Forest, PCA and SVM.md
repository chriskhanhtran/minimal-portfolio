
# Predict Breast Cancer with Random Forest, PCA and SVM


```python
__author__ = "Chris Tran"
__email__ = "tranduckhanh96@gmail.com"
__website__ = "chriskhanhtran.github.io"
```


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

# Part 1 - Introduction

In this project I am going to utilize Principal Components Analysis (PCA) to transform the breast cancer data set and then use Support Vector Machines model to predict whether a patient has breast cancer.

## Data Set Information
The [Breast Cancer Wisconsin (Diagnostic) Data Set](http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)) is obtained from UCI Machine Learning Repository. Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.

The dataset is also available in the Scikit Learn library. I will use Scikit Learn to import the dataset and explore its attributes.


```python
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
```

These are the elements of the data set:


```python
cancer.keys()
```




    dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])



The data set has 569 instances and 30 numeric variables:


```python
print(cancer.DESCR)
```

    .. _breast_cancer_dataset:
    
    Breast cancer wisconsin (diagnostic) dataset
    --------------------------------------------
    
    **Data Set Characteristics:**
    
        :Number of Instances: 569
    
        :Number of Attributes: 30 numeric, predictive attributes and the class
    
        :Attribute Information:
            - radius (mean of distances from center to points on the perimeter)
            - texture (standard deviation of gray-scale values)
            - perimeter
            - area
            - smoothness (local variation in radius lengths)
            - compactness (perimeter^2 / area - 1.0)
            - concavity (severity of concave portions of the contour)
            - concave points (number of concave portions of the contour)
            - symmetry 
            - fractal dimension ("coastline approximation" - 1)
    
            The mean, standard error, and "worst" or largest (mean of the three
            largest values) of these features were computed for each image,
            resulting in 30 features.  For instance, field 3 is Mean Radius, field
            13 is Radius SE, field 23 is Worst Radius.
    
            - class:
                    - WDBC-Malignant
                    - WDBC-Benign
    
        :Summary Statistics:
    
        ===================================== ====== ======
                                               Min    Max
        ===================================== ====== ======
        radius (mean):                        6.981  28.11
        texture (mean):                       9.71   39.28
        perimeter (mean):                     43.79  188.5
        area (mean):                          143.5  2501.0
        smoothness (mean):                    0.053  0.163
        compactness (mean):                   0.019  0.345
        concavity (mean):                     0.0    0.427
        concave points (mean):                0.0    0.201
        symmetry (mean):                      0.106  0.304
        fractal dimension (mean):             0.05   0.097
        radius (standard error):              0.112  2.873
        texture (standard error):             0.36   4.885
        perimeter (standard error):           0.757  21.98
        area (standard error):                6.802  542.2
        smoothness (standard error):          0.002  0.031
        compactness (standard error):         0.002  0.135
        concavity (standard error):           0.0    0.396
        concave points (standard error):      0.0    0.053
        symmetry (standard error):            0.008  0.079
        fractal dimension (standard error):   0.001  0.03
        radius (worst):                       7.93   36.04
        texture (worst):                      12.02  49.54
        perimeter (worst):                    50.41  251.2
        area (worst):                         185.2  4254.0
        smoothness (worst):                   0.071  0.223
        compactness (worst):                  0.027  1.058
        concavity (worst):                    0.0    1.252
        concave points (worst):               0.0    0.291
        symmetry (worst):                     0.156  0.664
        fractal dimension (worst):            0.055  0.208
        ===================================== ====== ======
    
        :Missing Attribute Values: None
    
        :Class Distribution: 212 - Malignant, 357 - Benign
    
        :Creator:  Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian
    
        :Donor: Nick Street
    
        :Date: November, 1995
    
    This is a copy of UCI ML Breast Cancer Wisconsin (Diagnostic) datasets.
    https://goo.gl/U2Uwz2
    
    Features are computed from a digitized image of a fine needle
    aspirate (FNA) of a breast mass.  They describe
    characteristics of the cell nuclei present in the image.
    
    Separating plane described above was obtained using
    Multisurface Method-Tree (MSM-T) [K. P. Bennett, "Decision Tree
    Construction Via Linear Programming." Proceedings of the 4th
    Midwest Artificial Intelligence and Cognitive Science Society,
    pp. 97-101, 1992], a classification method which uses linear
    programming to construct a decision tree.  Relevant features
    were selected using an exhaustive search in the space of 1-4
    features and 1-3 separating planes.
    
    The actual linear program used to obtain the separating plane
    in the 3-dimensional space is that described in:
    [K. P. Bennett and O. L. Mangasarian: "Robust Linear
    Programming Discrimination of Two Linearly Inseparable Sets",
    Optimization Methods and Software 1, 1992, 23-34].
    
    This database is also available through the UW CS ftp server:
    
    ftp ftp.cs.wisc.edu
    cd math-prog/cpo-dataset/machine-learn/WDBC/
    
    .. topic:: References
    
       - W.N. Street, W.H. Wolberg and O.L. Mangasarian. Nuclear feature extraction 
         for breast tumor diagnosis. IS&T/SPIE 1993 International Symposium on 
         Electronic Imaging: Science and Technology, volume 1905, pages 861-870,
         San Jose, CA, 1993.
       - O.L. Mangasarian, W.N. Street and W.H. Wolberg. Breast cancer diagnosis and 
         prognosis via linear programming. Operations Research, 43(4), pages 570-577, 
         July-August 1995.
       - W.H. Wolberg, W.N. Street, and O.L. Mangasarian. Machine learning techniques
         to diagnose breast cancer from fine-needle aspirates. Cancer Letters 77 (1994) 
         163-171.
    

# Part 2 - Discover

## Load the data

Let's take a look at feature variables:


```python
df_features = pd.DataFrame(cancer.data, columns = cancer.feature_names)
```


```python
df_features.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 569 entries, 0 to 568
    Data columns (total 30 columns):
    mean radius                569 non-null float64
    mean texture               569 non-null float64
    mean perimeter             569 non-null float64
    mean area                  569 non-null float64
    mean smoothness            569 non-null float64
    mean compactness           569 non-null float64
    mean concavity             569 non-null float64
    mean concave points        569 non-null float64
    mean symmetry              569 non-null float64
    mean fractal dimension     569 non-null float64
    radius error               569 non-null float64
    texture error              569 non-null float64
    perimeter error            569 non-null float64
    area error                 569 non-null float64
    smoothness error           569 non-null float64
    compactness error          569 non-null float64
    concavity error            569 non-null float64
    concave points error       569 non-null float64
    symmetry error             569 non-null float64
    fractal dimension error    569 non-null float64
    worst radius               569 non-null float64
    worst texture              569 non-null float64
    worst perimeter            569 non-null float64
    worst area                 569 non-null float64
    worst smoothness           569 non-null float64
    worst compactness          569 non-null float64
    worst concavity            569 non-null float64
    worst concave points       569 non-null float64
    worst symmetry             569 non-null float64
    worst fractal dimension    569 non-null float64
    dtypes: float64(30)
    memory usage: 133.4 KB
    

As the data is clean and has no missing value, the cleaning step can be skipped.

Target variable:


```python
cancer.target_names
```




    array(['malignant', 'benign'], dtype='<U9')




```python
df_target = pd.DataFrame(cancer.target, columns=['target'])
```


```python
df_target['target'].value_counts()
```




    1    357
    0    212
    Name: target, dtype: int64



According to the dataset's description, the distribution of the target variable is:
212 - Malignant, 357 - Benign. Thus, 'benign' and 'maglinant' are presented as 1 and 0, respectively.

Let's merge the features and target together:


```python
df = pd.concat([df_features, df_target], axis=1)
```

For the purpose of exploratory data analysis, I will transform the target data into text.


```python
df['target'] = df['target'].apply(lambda x: "Benign" if x==1 else "Malignant")
```


```python
df.head(5)
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
      <th>mean radius</th>
      <th>mean texture</th>
      <th>mean perimeter</th>
      <th>mean area</th>
      <th>mean smoothness</th>
      <th>mean compactness</th>
      <th>mean concavity</th>
      <th>mean concave points</th>
      <th>mean symmetry</th>
      <th>mean fractal dimension</th>
      <th>...</th>
      <th>worst texture</th>
      <th>worst perimeter</th>
      <th>worst area</th>
      <th>worst smoothness</th>
      <th>worst compactness</th>
      <th>worst concavity</th>
      <th>worst concave points</th>
      <th>worst symmetry</th>
      <th>worst fractal dimension</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>0.2419</td>
      <td>0.07871</td>
      <td>...</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
      <td>Malignant</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>0.1812</td>
      <td>0.05667</td>
      <td>...</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
      <td>Malignant</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>0.2069</td>
      <td>0.05999</td>
      <td>...</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>1709.0</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
      <td>Malignant</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.2414</td>
      <td>0.10520</td>
      <td>0.2597</td>
      <td>0.09744</td>
      <td>...</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>567.7</td>
      <td>0.2098</td>
      <td>0.8663</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
      <td>Malignant</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.1980</td>
      <td>0.10430</td>
      <td>0.1809</td>
      <td>0.05883</td>
      <td>...</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>1575.0</td>
      <td>0.1374</td>
      <td>0.2050</td>
      <td>0.4000</td>
      <td>0.1625</td>
      <td>0.2364</td>
      <td>0.07678</td>
      <td>Malignant</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>




```python
df.describe()
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
      <th>mean radius</th>
      <th>mean texture</th>
      <th>mean perimeter</th>
      <th>mean area</th>
      <th>mean smoothness</th>
      <th>mean compactness</th>
      <th>mean concavity</th>
      <th>mean concave points</th>
      <th>mean symmetry</th>
      <th>mean fractal dimension</th>
      <th>...</th>
      <th>worst radius</th>
      <th>worst texture</th>
      <th>worst perimeter</th>
      <th>worst area</th>
      <th>worst smoothness</th>
      <th>worst compactness</th>
      <th>worst concavity</th>
      <th>worst concave points</th>
      <th>worst symmetry</th>
      <th>worst fractal dimension</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>...</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>14.127292</td>
      <td>19.289649</td>
      <td>91.969033</td>
      <td>654.889104</td>
      <td>0.096360</td>
      <td>0.104341</td>
      <td>0.088799</td>
      <td>0.048919</td>
      <td>0.181162</td>
      <td>0.062798</td>
      <td>...</td>
      <td>16.269190</td>
      <td>25.677223</td>
      <td>107.261213</td>
      <td>880.583128</td>
      <td>0.132369</td>
      <td>0.254265</td>
      <td>0.272188</td>
      <td>0.114606</td>
      <td>0.290076</td>
      <td>0.083946</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.524049</td>
      <td>4.301036</td>
      <td>24.298981</td>
      <td>351.914129</td>
      <td>0.014064</td>
      <td>0.052813</td>
      <td>0.079720</td>
      <td>0.038803</td>
      <td>0.027414</td>
      <td>0.007060</td>
      <td>...</td>
      <td>4.833242</td>
      <td>6.146258</td>
      <td>33.602542</td>
      <td>569.356993</td>
      <td>0.022832</td>
      <td>0.157336</td>
      <td>0.208624</td>
      <td>0.065732</td>
      <td>0.061867</td>
      <td>0.018061</td>
    </tr>
    <tr>
      <th>min</th>
      <td>6.981000</td>
      <td>9.710000</td>
      <td>43.790000</td>
      <td>143.500000</td>
      <td>0.052630</td>
      <td>0.019380</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.106000</td>
      <td>0.049960</td>
      <td>...</td>
      <td>7.930000</td>
      <td>12.020000</td>
      <td>50.410000</td>
      <td>185.200000</td>
      <td>0.071170</td>
      <td>0.027290</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.156500</td>
      <td>0.055040</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>11.700000</td>
      <td>16.170000</td>
      <td>75.170000</td>
      <td>420.300000</td>
      <td>0.086370</td>
      <td>0.064920</td>
      <td>0.029560</td>
      <td>0.020310</td>
      <td>0.161900</td>
      <td>0.057700</td>
      <td>...</td>
      <td>13.010000</td>
      <td>21.080000</td>
      <td>84.110000</td>
      <td>515.300000</td>
      <td>0.116600</td>
      <td>0.147200</td>
      <td>0.114500</td>
      <td>0.064930</td>
      <td>0.250400</td>
      <td>0.071460</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>13.370000</td>
      <td>18.840000</td>
      <td>86.240000</td>
      <td>551.100000</td>
      <td>0.095870</td>
      <td>0.092630</td>
      <td>0.061540</td>
      <td>0.033500</td>
      <td>0.179200</td>
      <td>0.061540</td>
      <td>...</td>
      <td>14.970000</td>
      <td>25.410000</td>
      <td>97.660000</td>
      <td>686.500000</td>
      <td>0.131300</td>
      <td>0.211900</td>
      <td>0.226700</td>
      <td>0.099930</td>
      <td>0.282200</td>
      <td>0.080040</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>15.780000</td>
      <td>21.800000</td>
      <td>104.100000</td>
      <td>782.700000</td>
      <td>0.105300</td>
      <td>0.130400</td>
      <td>0.130700</td>
      <td>0.074000</td>
      <td>0.195700</td>
      <td>0.066120</td>
      <td>...</td>
      <td>18.790000</td>
      <td>29.720000</td>
      <td>125.400000</td>
      <td>1084.000000</td>
      <td>0.146000</td>
      <td>0.339100</td>
      <td>0.382900</td>
      <td>0.161400</td>
      <td>0.317900</td>
      <td>0.092080</td>
    </tr>
    <tr>
      <th>max</th>
      <td>28.110000</td>
      <td>39.280000</td>
      <td>188.500000</td>
      <td>2501.000000</td>
      <td>0.163400</td>
      <td>0.345400</td>
      <td>0.426800</td>
      <td>0.201200</td>
      <td>0.304000</td>
      <td>0.097440</td>
      <td>...</td>
      <td>36.040000</td>
      <td>49.540000</td>
      <td>251.200000</td>
      <td>4254.000000</td>
      <td>0.222600</td>
      <td>1.058000</td>
      <td>1.252000</td>
      <td>0.291000</td>
      <td>0.663800</td>
      <td>0.207500</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 30 columns</p>
</div>



## Exploratory Data Analysis


```python
# Set style
sns.set_style('darkgrid')
```

### Count plot of each diagnosis


```python
df['target'].value_counts()
```




    Benign       357
    Malignant    212
    Name: target, dtype: int64




```python
plt.figure(figsize=(8, 6))
sns.countplot(df['target'])
plt.xlabel("Diagnosis")
plt.title("Count Plot of Diagnosis")
```




    Text(0.5, 1.0, 'Count Plot of Diagnosis')




![png](output_31_1.png)


### Distribution of features

Now I will take a look at the distribution of each feature and see how they are different between 'benign' and 'malignant'. To see the distribution of multiple variables, we can use violin plot, swarm plot or box plot. Let's try each of these plots.

To visualize distributions of multiple features in one figure, first I need to standardize the data:


```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(df_features)

features_scaled = scaler.transform(df_features)
features_scaled = pd.DataFrame(data=features_scaled,
                               columns=df_features.columns)

df_scaled = pd.concat([features_scaled, df['target']], axis=1)
```

Then I "unpivot" the dataframe from wide format to long format using pd.melt function:


```python
df_scaled_melt = pd.melt(df_scaled, id_vars='target',
                         var_name='features', value_name='value')
```


```python
df_scaled_melt.head(3)
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
      <th>target</th>
      <th>features</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Malignant</td>
      <td>mean radius</td>
      <td>1.097064</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Malignant</td>
      <td>mean radius</td>
      <td>1.829821</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Malignant</td>
      <td>mean radius</td>
      <td>1.579888</td>
    </tr>
  </tbody>
</table>
</div>



There are 30 features so I will create a violin plot, a swarm plot and a box plot for each batch of 10 features:


```python
def violin_plot(features, name):
    """
    This function creates violin plots of features given in the argument.
    """
    # Create query
    query = ''
    for x in features:
        query += "features == '"+ str(x) +"' or "
    query = query[0:-4]
    
    # Create data for visualization
    data = df_scaled_melt.query(query)
    
    # Plot figure
    plt.figure(figsize=(12,6))
    sns.violinplot(x='features', y='value', hue='target', data=data, split=True, inner="quart")
    plt.xticks(rotation=45)
    plt.title(name)
    plt.xlabel("Features")
    plt.ylabel("Standardize Value")
    
def swarm_plot(features, name):
    """
    This function creates swarm plots of features given in the argument.
    """
    # Create query
    query = ''
    for x in features:
        query += "features == '"+ str(x) +"' or "
    query = query[0:-4]
    
    # Create data for visualization
    data = df_scaled_melt.query(query)
    
    # Plot figure
    plt.figure(figsize=(12,6))
    sns.swarmplot(x='features', y='value', hue='target', data=data)
    plt.xticks(rotation=45)
    plt.title(name)
    plt.xlabel("Features")
    plt.ylabel("Standardize Value")
    
def box_plot(features, name):
    """
    This function creates box plots of features given in the argument.
    """
    # Create query
    query = ''
    for x in features:
        query += "features == '"+ str(x) +"' or "
    query = query[0:-4]
    
    # Create data for visualization
    data = df_scaled_melt.query(query)
    
    # Plot figure
    plt.figure(figsize=(12,6))
    sns.boxplot(x='features', y='value', hue='target', data=data)
    plt.xticks(rotation=45)
    plt.title(name)
    plt.xlabel("Features")
    plt.ylabel("Standardize Value")    
```


```python
violin_plot(df.columns[0:10], "Violin Plot of the First 10 Features")
```


![png](output_40_0.png)



```python
swarm_plot(df.columns[10:20], "Swarm Plot of the Next 10 Features")
```


![png](output_41_0.png)



```python
box_plot(df.columns[20:30], "Box Plot of the Last 10 Features")
```


![png](output_42_0.png)


The violin plot shows the distribution of each feature by target variable. The classification becomes clear in the swarm plot. Finally, the box plots are useful in comparing median and detecing outliers.

From these plots we can draw some insights from the data:
- The median of some features are very different between 'malignant' and 'benign'. This seperation can be seen clearly in the box plots. They can be very good features for classification. For examples: `mean radius`, `mean area`, `mean concave points`, `worst radius`, `worst perimeter`, `worst area`, `worst concave points`.

- However, there are distributions looking similar between 'malignant' and 'benign'. For examples: `mean smoothness`, `mean symmetry`, `mean fractual dimension`, `smoothness error`. These features are weak in classifying data.

- Some features have similar distributions, thus might be highly correlated with each other. For example: `mean perimeter` vs. `mean area`, `mean concavity` vs. `mean concave points`, and `worst symmetry` vs. `worst fractal dimension`. We should not include these hightly correlated varibles in our predicting model.

### Correlation

As discussed above, some dependent variables in the dataset might be highly correlated with each other. Let's explore the correlation of three examples above.


```python
def correlation(var):
    """
    1. Print correlation
    2. Create jointplot
    """
    # Print correlation
    print("Correlation: ", df[[var[0], var[1]]].corr().iloc[1, 0])

    # Create jointplot
    plt.figure(figsize=(6, 6))
    sns.jointplot(df[(var[0])], df[(var[1])], kind='reg')
```


```python
correlation(['mean perimeter', 'mean area'])
```

    Correlation:  0.9865068039913906
    

    C:\ProgramData\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    


    <Figure size 432x432 with 0 Axes>



![png](output_47_3.png)



```python
correlation(['mean concavity', 'mean concave points'])
```

    Correlation:  0.9213910263788593
    


    <Figure size 432x432 with 0 Axes>



![png](output_48_2.png)



```python
correlation(['worst symmetry', 'worst fractal dimension'])
```

    Correlation:  0.5378482062536076
    


    <Figure size 432x432 with 0 Axes>



![png](output_49_2.png)


Two pairs of 3 examples are actually highly correlated. Let's create a heatmap to see the overall picture on correlation.


```python
corr_mat = df.corr()
```


```python
plt.figure(figsize=(15, 10))
sns.heatmap(corr_mat, annot=True, fmt='.1f', cmap='viridis_r')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2a114086ac8>




![png](output_52_1.png)


From the heat map, we can see that many variables in the dataset are highly correlated. What are variables having correlation greater than 0.8?


```python
plt.figure(figsize=(15, 10))
sns.heatmap(corr_mat[corr_mat > 0.8], annot=True, fmt='.1f', cmap='coolwarm')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2a114086358>




![png](output_54_1.png)


Well, we have some work to do with feature selection.

# Part 3 - Create Model

## 1. Feature Selection and Random Forest Classifier

### Feature selection

I will use Univariate Feature Selection [(sklearn.feature_selection.SelectKBest)](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest) to choose 5 features with k highest scores. The number of features to be selected could be 2, 10 or 15. I choose 5 because from the heatmap I could see about 5 groups of features that are highly correlated.


```python
from sklearn.feature_selection import SelectKBest, chi2
```


```python
feature_selection = SelectKBest(chi2, k=5)
```


```python
feature_selection.fit(df_features, df_target)
```




    SelectKBest(k=5, score_func=<function chi2 at 0x000002A113E39EA0>)




```python
selected_features = df_features.columns[feature_selection.get_support()]
print("The five selected features are: ", list(selected_features))
```

    The five selected features are:  ['mean perimeter', 'mean area', 'area error', 'worst perimeter', 'worst area']
    


```python
X = pd.DataFrame(feature_selection.transform(df_features),
                 columns=selected_features)
```


```python
X.head()
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
      <th>mean perimeter</th>
      <th>mean area</th>
      <th>area error</th>
      <th>worst perimeter</th>
      <th>worst area</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>122.80</td>
      <td>1001.0</td>
      <td>153.40</td>
      <td>184.60</td>
      <td>2019.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>132.90</td>
      <td>1326.0</td>
      <td>74.08</td>
      <td>158.80</td>
      <td>1956.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>130.00</td>
      <td>1203.0</td>
      <td>94.03</td>
      <td>152.50</td>
      <td>1709.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>77.58</td>
      <td>386.1</td>
      <td>27.23</td>
      <td>98.87</td>
      <td>567.7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>135.10</td>
      <td>1297.0</td>
      <td>94.44</td>
      <td>152.20</td>
      <td>1575.0</td>
    </tr>
  </tbody>
</table>
</div>



Let's create a pairplot to see how different these features are in 'malignant' and in 'benign'.


```python
sns.pairplot(pd.concat([X, df['target']], axis=1), hue='target')
```

    C:\ProgramData\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    




    <seaborn.axisgrid.PairGrid at 0x2a113e3ac50>




![png](output_67_2.png)


### Train test split


```python
from sklearn.model_selection import train_test_split
y = df_target['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
```

### Random Forest Classifier


```python
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
```

### Model evaluation


```python
from sklearn.metrics import confusion_matrix, classification_report
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n")
print("Classification Report:\n",classification_report(y_test, y_pred))
```

    Confusion Matrix:
     [[ 63   4]
     [  3 118]]
    
    
    Classification Report:
                   precision    recall  f1-score   support
    
               0       0.95      0.94      0.95        67
               1       0.97      0.98      0.97       121
    
       micro avg       0.96      0.96      0.96       188
       macro avg       0.96      0.96      0.96       188
    weighted avg       0.96      0.96      0.96       188
    
    

The accuracy rate is approximately 97%. The model only makes 5 wrong predictions out of 188. Our chosen features are pretty good in identify cancer.

## 2. PCA and SVM

### Feature extraction using principal component analysis (PCA)


```python
from sklearn.decomposition import PCA
```

Principal component analysis (PCA) performs linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space.

PCA transforms the data into features that explain the most variance in our data.

Fore better performance of PCA, we first need to scale our data so that each features has a single unit variance. I have done this step in EDA.


```python
features_scaled.head(5)
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
      <th>mean radius</th>
      <th>mean texture</th>
      <th>mean perimeter</th>
      <th>mean area</th>
      <th>mean smoothness</th>
      <th>mean compactness</th>
      <th>mean concavity</th>
      <th>mean concave points</th>
      <th>mean symmetry</th>
      <th>mean fractal dimension</th>
      <th>...</th>
      <th>worst radius</th>
      <th>worst texture</th>
      <th>worst perimeter</th>
      <th>worst area</th>
      <th>worst smoothness</th>
      <th>worst compactness</th>
      <th>worst concavity</th>
      <th>worst concave points</th>
      <th>worst symmetry</th>
      <th>worst fractal dimension</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.097064</td>
      <td>-2.073335</td>
      <td>1.269934</td>
      <td>0.984375</td>
      <td>1.568466</td>
      <td>3.283515</td>
      <td>2.652874</td>
      <td>2.532475</td>
      <td>2.217515</td>
      <td>2.255747</td>
      <td>...</td>
      <td>1.886690</td>
      <td>-1.359293</td>
      <td>2.303601</td>
      <td>2.001237</td>
      <td>1.307686</td>
      <td>2.616665</td>
      <td>2.109526</td>
      <td>2.296076</td>
      <td>2.750622</td>
      <td>1.937015</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.829821</td>
      <td>-0.353632</td>
      <td>1.685955</td>
      <td>1.908708</td>
      <td>-0.826962</td>
      <td>-0.487072</td>
      <td>-0.023846</td>
      <td>0.548144</td>
      <td>0.001392</td>
      <td>-0.868652</td>
      <td>...</td>
      <td>1.805927</td>
      <td>-0.369203</td>
      <td>1.535126</td>
      <td>1.890489</td>
      <td>-0.375612</td>
      <td>-0.430444</td>
      <td>-0.146749</td>
      <td>1.087084</td>
      <td>-0.243890</td>
      <td>0.281190</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.579888</td>
      <td>0.456187</td>
      <td>1.566503</td>
      <td>1.558884</td>
      <td>0.942210</td>
      <td>1.052926</td>
      <td>1.363478</td>
      <td>2.037231</td>
      <td>0.939685</td>
      <td>-0.398008</td>
      <td>...</td>
      <td>1.511870</td>
      <td>-0.023974</td>
      <td>1.347475</td>
      <td>1.456285</td>
      <td>0.527407</td>
      <td>1.082932</td>
      <td>0.854974</td>
      <td>1.955000</td>
      <td>1.152255</td>
      <td>0.201391</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.768909</td>
      <td>0.253732</td>
      <td>-0.592687</td>
      <td>-0.764464</td>
      <td>3.283553</td>
      <td>3.402909</td>
      <td>1.915897</td>
      <td>1.451707</td>
      <td>2.867383</td>
      <td>4.910919</td>
      <td>...</td>
      <td>-0.281464</td>
      <td>0.133984</td>
      <td>-0.249939</td>
      <td>-0.550021</td>
      <td>3.394275</td>
      <td>3.893397</td>
      <td>1.989588</td>
      <td>2.175786</td>
      <td>6.046041</td>
      <td>4.935010</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.750297</td>
      <td>-1.151816</td>
      <td>1.776573</td>
      <td>1.826229</td>
      <td>0.280372</td>
      <td>0.539340</td>
      <td>1.371011</td>
      <td>1.428493</td>
      <td>-0.009560</td>
      <td>-0.562450</td>
      <td>...</td>
      <td>1.298575</td>
      <td>-1.466770</td>
      <td>1.338539</td>
      <td>1.220724</td>
      <td>0.220556</td>
      <td>-0.313395</td>
      <td>0.613179</td>
      <td>0.729259</td>
      <td>-0.868353</td>
      <td>-0.397100</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>




```python
X_scaled = features_scaled
```

It's difficult to visualize high-dimensional data like our original data. I will use PCA to find two most principal components and visualize the data in this two-dimensional space.


```python
pca = PCA(n_components=2)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)
```

Let's visualize our data based on these two principle components:


```python
plt.figure(figsize=(8,8))
sns.scatterplot(X_pca[:,0], X_pca[:,1], hue=df['target'])
plt.title("PCA")
plt.xlabel("First Principal Component")
plt.xlabel("Second Principal Component")
```




    Text(0.5, 0, 'Second Principal Component')




![png](output_84_1.png)


We can use the two principal components to clearly separace our data between *Malignant* and *Benign*.

### Train Test Split


```python
X = X_pca
y = df_target['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
```

### Support Vector Machines (SVM)


```python
from sklearn.svm import SVC
```

After transforming the data into 2 principal components, I will use SVM model to predict cancer.

**GridSearch**


```python
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)

grid.fit(X_train, y_train)
```

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_split.py:2053: FutureWarning: You should specify a value for 'cv' instead of relying on the default value. The default value will change from 3 to 5 in version 0.22.
      warnings.warn(CV_WARNING, FutureWarning)
    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.0s remaining:    0.0s
    [Parallel(n_jobs=1)]: Done   2 out of   2 | elapsed:    0.0s remaining:    0.0s
    

    Fitting 3 folds for each of 25 candidates, totalling 75 fits
    [CV] C=0.1, gamma=1, kernel=rbf ......................................
    [CV] ...... C=0.1, gamma=1, kernel=rbf, score=0.8203125, total=   0.0s
    [CV] C=0.1, gamma=1, kernel=rbf ......................................
    [CV]  C=0.1, gamma=1, kernel=rbf, score=0.84251968503937, total=   0.0s
    [CV] C=0.1, gamma=1, kernel=rbf ......................................
    [CV]  C=0.1, gamma=1, kernel=rbf, score=0.8253968253968254, total=   0.0s
    [CV] C=0.1, gamma=0.1, kernel=rbf ....................................
    [CV] .... C=0.1, gamma=0.1, kernel=rbf, score=0.9453125, total=   0.0s
    [CV] C=0.1, gamma=0.1, kernel=rbf ....................................
    [CV]  C=0.1, gamma=0.1, kernel=rbf, score=0.905511811023622, total=   0.0s
    [CV] C=0.1, gamma=0.1, kernel=rbf ....................................
    [CV]  C=0.1, gamma=0.1, kernel=rbf, score=0.9365079365079365, total=   0.0s
    [CV] C=0.1, gamma=0.01, kernel=rbf ...................................
    [CV] ... C=0.1, gamma=0.01, kernel=rbf, score=0.8984375, total=   0.0s
    [CV] C=0.1, gamma=0.01, kernel=rbf ...................................
    [CV]  C=0.1, gamma=0.01, kernel=rbf, score=0.9212598425196851, total=   0.0s
    [CV] C=0.1, gamma=0.01, kernel=rbf ...................................
    [CV]  C=0.1, gamma=0.01, kernel=rbf, score=0.9285714285714286, total=   0.0s
    [CV] C=0.1, gamma=0.001, kernel=rbf ..................................
    [CV] ... C=0.1, gamma=0.001, kernel=rbf, score=0.640625, total=   0.0s
    [CV] C=0.1, gamma=0.001, kernel=rbf ..................................
    [CV]  C=0.1, gamma=0.001, kernel=rbf, score=0.6377952755905512, total=   0.0s
    [CV] C=0.1, gamma=0.001, kernel=rbf ..................................
    [CV]  C=0.1, gamma=0.001, kernel=rbf, score=0.6587301587301587, total=   0.0s
    [CV] C=0.1, gamma=0.0001, kernel=rbf .................................
    [CV] . C=0.1, gamma=0.0001, kernel=rbf, score=0.6171875, total=   0.0s
    [CV] C=0.1, gamma=0.0001, kernel=rbf .................................
    [CV]  C=0.1, gamma=0.0001, kernel=rbf, score=0.6220472440944882, total=   0.0s
    [CV] C=0.1, gamma=0.0001, kernel=rbf .................................
    [CV]  C=0.1, gamma=0.0001, kernel=rbf, score=0.6190476190476191, total=   0.0s
    [CV] C=1, gamma=1, kernel=rbf ........................................
    [CV] ......... C=1, gamma=1, kernel=rbf, score=0.921875, total=   0.0s
    [CV] C=1, gamma=1, kernel=rbf ........................................
    [CV]  C=1, gamma=1, kernel=rbf, score=0.8976377952755905, total=   0.0s
    [CV] C=1, gamma=1, kernel=rbf ........................................
    [CV]  C=1, gamma=1, kernel=rbf, score=0.9285714285714286, total=   0.0s
    [CV] C=1, gamma=0.1, kernel=rbf ......................................
    [CV] ...... C=1, gamma=0.1, kernel=rbf, score=0.9296875, total=   0.0s
    [CV] C=1, gamma=0.1, kernel=rbf ......................................
    [CV]  C=1, gamma=0.1, kernel=rbf, score=0.937007874015748, total=   0.0s
    [CV] C=1, gamma=0.1, kernel=rbf ......................................
    [CV]  C=1, gamma=0.1, kernel=rbf, score=0.9285714285714286, total=   0.0s
    [CV] C=1, gamma=0.01, kernel=rbf .....................................
    [CV] ........ C=1, gamma=0.01, kernel=rbf, score=0.9375, total=   0.0s
    [CV] C=1, gamma=0.01, kernel=rbf .....................................
    [CV]  C=1, gamma=0.01, kernel=rbf, score=0.9291338582677166, total=   0.0s
    [CV] C=1, gamma=0.01, kernel=rbf .....................................
    [CV]  C=1, gamma=0.01, kernel=rbf, score=0.9365079365079365, total=   0.0s
    [CV] C=1, gamma=0.001, kernel=rbf ....................................
    [CV] ...... C=1, gamma=0.001, kernel=rbf, score=0.90625, total=   0.0s
    [CV] C=1, gamma=0.001, kernel=rbf ....................................
    [CV]  C=1, gamma=0.001, kernel=rbf, score=0.937007874015748, total=   0.0s
    [CV] C=1, gamma=0.001, kernel=rbf ....................................
    [CV]  C=1, gamma=0.001, kernel=rbf, score=0.9285714285714286, total=   0.0s
    [CV] C=1, gamma=0.0001, kernel=rbf ...................................
    [CV] ..... C=1, gamma=0.0001, kernel=rbf, score=0.65625, total=   0.0s
    [CV] C=1, gamma=0.0001, kernel=rbf ...................................
    [CV]  C=1, gamma=0.0001, kernel=rbf, score=0.6535433070866141, total=   0.0s
    [CV] C=1, gamma=0.0001, kernel=rbf ...................................
    [CV]  C=1, gamma=0.0001, kernel=rbf, score=0.6587301587301587, total=   0.0s
    [CV] C=10, gamma=1, kernel=rbf .......................................
    [CV] ........ C=10, gamma=1, kernel=rbf, score=0.921875, total=   0.0s
    [CV] C=10, gamma=1, kernel=rbf .......................................
    [CV]  C=10, gamma=1, kernel=rbf, score=0.8976377952755905, total=   0.0s
    [CV] C=10, gamma=1, kernel=rbf .......................................
    [CV]  C=10, gamma=1, kernel=rbf, score=0.9444444444444444, total=   0.0s
    [CV] C=10, gamma=0.1, kernel=rbf .....................................
    [CV] ..... C=10, gamma=0.1, kernel=rbf, score=0.9296875, total=   0.0s
    [CV] C=10, gamma=0.1, kernel=rbf .....................................
    [CV]  C=10, gamma=0.1, kernel=rbf, score=0.937007874015748, total=   0.0s
    [CV] C=10, gamma=0.1, kernel=rbf .....................................
    [CV]  C=10, gamma=0.1, kernel=rbf, score=0.9285714285714286, total=   0.0s
    [CV] C=10, gamma=0.01, kernel=rbf ....................................
    [CV] .... C=10, gamma=0.01, kernel=rbf, score=0.9453125, total=   0.0s
    [CV] C=10, gamma=0.01, kernel=rbf ....................................
    [CV]  C=10, gamma=0.01, kernel=rbf, score=0.9291338582677166, total=   0.0s
    [CV] C=10, gamma=0.01, kernel=rbf ....................................
    [CV]  C=10, gamma=0.01, kernel=rbf, score=0.9285714285714286, total=   0.0s
    [CV] C=10, gamma=0.001, kernel=rbf ...................................
    [CV] ... C=10, gamma=0.001, kernel=rbf, score=0.9296875, total=   0.0s
    [CV] C=10, gamma=0.001, kernel=rbf ...................................
    [CV]  C=10, gamma=0.001, kernel=rbf, score=0.937007874015748, total=   0.0s
    [CV] C=10, gamma=0.001, kernel=rbf ...................................
    [CV]  C=10, gamma=0.001, kernel=rbf, score=0.9523809523809523, total=   0.0s
    [CV] C=10, gamma=0.0001, kernel=rbf ..................................
    [CV] .... C=10, gamma=0.0001, kernel=rbf, score=0.90625, total=   0.0s
    [CV] C=10, gamma=0.0001, kernel=rbf ..................................
    [CV]  C=10, gamma=0.0001, kernel=rbf, score=0.937007874015748, total=   0.0s
    [CV] C=10, gamma=0.0001, kernel=rbf ..................................
    [CV]  C=10, gamma=0.0001, kernel=rbf, score=0.9285714285714286, total=   0.0s
    [CV] C=100, gamma=1, kernel=rbf ......................................
    [CV] ....... C=100, gamma=1, kernel=rbf, score=0.921875, total=   0.0s
    [CV] C=100, gamma=1, kernel=rbf ......................................
    [CV]  C=100, gamma=1, kernel=rbf, score=0.889763779527559, total=   0.0s
    [CV] C=100, gamma=1, kernel=rbf ......................................
    [CV]  C=100, gamma=1, kernel=rbf, score=0.9206349206349206, total=   0.0s
    [CV] C=100, gamma=0.1, kernel=rbf ....................................
    [CV] ....... C=100, gamma=0.1, kernel=rbf, score=0.9375, total=   0.0s
    [CV] C=100, gamma=0.1, kernel=rbf ....................................
    [CV]  C=100, gamma=0.1, kernel=rbf, score=0.9133858267716536, total=   0.0s
    [CV] C=100, gamma=0.1, kernel=rbf ....................................
    [CV]  C=100, gamma=0.1, kernel=rbf, score=0.9206349206349206, total=   0.0s
    [CV] C=100, gamma=0.01, kernel=rbf ...................................
    [CV] ... C=100, gamma=0.01, kernel=rbf, score=0.9296875, total=   0.0s
    [CV] C=100, gamma=0.01, kernel=rbf ...................................
    [CV]  C=100, gamma=0.01, kernel=rbf, score=0.937007874015748, total=   0.0s
    [CV] C=100, gamma=0.01, kernel=rbf ...................................
    [CV]  C=100, gamma=0.01, kernel=rbf, score=0.9444444444444444, total=   0.0s
    [CV] C=100, gamma=0.001, kernel=rbf ..................................
    [CV] .. C=100, gamma=0.001, kernel=rbf, score=0.9296875, total=   0.0s
    [CV] C=100, gamma=0.001, kernel=rbf ..................................
    [CV]  C=100, gamma=0.001, kernel=rbf, score=0.937007874015748, total=   0.0s
    [CV] C=100, gamma=0.001, kernel=rbf ..................................
    [CV]  C=100, gamma=0.001, kernel=rbf, score=0.9523809523809523, total=   0.0s
    [CV] C=100, gamma=0.0001, kernel=rbf .................................
    [CV] .... C=100, gamma=0.0001, kernel=rbf, score=0.9375, total=   0.0s
    [CV] C=100, gamma=0.0001, kernel=rbf .................................
    [CV]  C=100, gamma=0.0001, kernel=rbf, score=0.937007874015748, total=   0.0s
    [CV] C=100, gamma=0.0001, kernel=rbf .................................
    [CV]  C=100, gamma=0.0001, kernel=rbf, score=0.9444444444444444, total=   0.0s
    [CV] C=1000, gamma=1, kernel=rbf .....................................
    [CV] ..... C=1000, gamma=1, kernel=rbf, score=0.8828125, total=   0.0s
    [CV] C=1000, gamma=1, kernel=rbf .....................................
    [CV]  C=1000, gamma=1, kernel=rbf, score=0.889763779527559, total=   0.0s
    [CV] C=1000, gamma=1, kernel=rbf .....................................
    [CV]  C=1000, gamma=1, kernel=rbf, score=0.8571428571428571, total=   0.0s
    [CV] C=1000, gamma=0.1, kernel=rbf ...................................
    [CV] ... C=1000, gamma=0.1, kernel=rbf, score=0.9453125, total=   0.0s
    [CV] C=1000, gamma=0.1, kernel=rbf ...................................
    [CV]  C=1000, gamma=0.1, kernel=rbf, score=0.905511811023622, total=   0.0s
    [CV] C=1000, gamma=0.1, kernel=rbf ...................................
    [CV]  C=1000, gamma=0.1, kernel=rbf, score=0.9444444444444444, total=   0.0s
    [CV] C=1000, gamma=0.01, kernel=rbf ..................................
    [CV] ... C=1000, gamma=0.01, kernel=rbf, score=0.921875, total=   0.0s
    [CV] C=1000, gamma=0.01, kernel=rbf ..................................
    [CV]  C=1000, gamma=0.01, kernel=rbf, score=0.937007874015748, total=   0.0s
    [CV] C=1000, gamma=0.01, kernel=rbf ..................................
    [CV]  C=1000, gamma=0.01, kernel=rbf, score=0.9523809523809523, total=   0.0s
    [CV] C=1000, gamma=0.001, kernel=rbf .................................
    [CV] . C=1000, gamma=0.001, kernel=rbf, score=0.9609375, total=   0.0s
    [CV] C=1000, gamma=0.001, kernel=rbf .................................
    [CV]  C=1000, gamma=0.001, kernel=rbf, score=0.9291338582677166, total=   0.0s
    [CV] C=1000, gamma=0.001, kernel=rbf .................................
    [CV]  C=1000, gamma=0.001, kernel=rbf, score=0.9444444444444444, total=   0.0s
    [CV] C=1000, gamma=0.0001, kernel=rbf ................................
    [CV]  C=1000, gamma=0.0001, kernel=rbf, score=0.9296875, total=   0.0s
    [CV] C=1000, gamma=0.0001, kernel=rbf ................................
    [CV]  C=1000, gamma=0.0001, kernel=rbf, score=0.937007874015748, total=   0.0s
    [CV] C=1000, gamma=0.0001, kernel=rbf ................................
    [CV]  C=1000, gamma=0.0001, kernel=rbf, score=0.9444444444444444, total=   0.0s
    

    [Parallel(n_jobs=1)]: Done  75 out of  75 | elapsed:    0.3s finished
    




    GridSearchCV(cv='warn', error_score='raise-deprecating',
           estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
      kernel='rbf', max_iter=-1, probability=False, random_state=None,
      shrinking=True, tol=0.001, verbose=False),
           fit_params=None, iid='warn', n_jobs=None,
           param_grid={'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=3)




```python
grid.best_params_
```




    {'C': 1000, 'gamma': 0.001, 'kernel': 'rbf'}




```python
y_pred = grid.predict(X_test)
```

### Model Evaluation


```python
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n")
print("Classification Report:\n",classification_report(y_test, y_pred))
```

    Confusion Matrix:
     [[ 62   5]
     [  5 116]]
    
    
    Classification Report:
                   precision    recall  f1-score   support
    
               0       0.93      0.93      0.93        67
               1       0.96      0.96      0.96       121
    
       micro avg       0.95      0.95      0.95       188
       macro avg       0.94      0.94      0.94       188
    weighted avg       0.95      0.95      0.95       188
    
    

The accuracy rate is 95%. This model made a bit more wrong predictions than the Random Forest model. However, with PCA, we can reduce the number of dimensions in our data.

# Part 4 - Conclusion

In the first part of this project, I performed exploratory data analysis to better understand each of 30 original features and how they might be associated with cancer.

Next, I selected 5 best features for my model using univariate feature selection, and performed Random Forst classifier. The accuracy rate of this model is 97%.

In addition, I used PCA to find the two principle components and created visualization based on these two variables. The visualization shows that with only two variables, we can clearly separate the data between cancer and no cancer. Finally, I preformed Support Vector Machines model to predict cancer based on PCA. The accuracy rate for this SVM model is 95%.

In fact, this data set is quite easy for machine learning models to classify. However, my purpose of doing this project is to learn how to mine the data by exploring each feature, select features for my model, and perform various machine learning models.

I hope you enjoy this project. If you have any questions, please contact me at tranduckhanh96@gmail.com. Thank you for reading!
