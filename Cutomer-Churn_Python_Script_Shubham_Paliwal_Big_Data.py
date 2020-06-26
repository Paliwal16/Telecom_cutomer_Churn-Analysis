#!/usr/bin/env python
# coding: utf-8

# ## Shubham Paliwal
# ## Big-Data using Spark
# ## Analysis on Churn.csv Dataset

# ## __________________________________________
# ## Telecom Customer Churn
# ### Introduction
# 
# #### I am going to use this Telecom Customer dataset to predict the churn with the Random Forest, Decision Tree and Logistic Regression algorithm.
# 

# ### Setting Dictionary (path) from cloudera:

# In[1]:


import findspark
findspark.init('/opt/cloudera/parcels/SPARK2-2.4.0.cloudera2-1.cdh5.13.3.p0.1041012/lib/spark2/')


# In[2]:


from pyspark.sql import SparkSession


# ###  Setting directory to read our dataset into spark from Hive or Hbase.

# In[3]:


# Quetion - 1)
spark_2 = SparkSession.builder.appName('SparkSession2').config('spark.sql.warehouse.dir', 'hdfs://masternode:8020/user/hive/warehouse/').enableHiveSupport().getOrCreate()


# In[4]:


df = spark_2.read.csv("hdfs://masternode:8020/user/bdhlabnova3/Churn.csv", header=True, inferSchema=True)


# ### As you can see, we have successfully imported our data from hive (Hbase), Now we will begin with the very first step to build our model  - EDA (Exploratory Data Analysis)
# #### To take brief view of data in spark we use .Show() command:

# In[5]:


df.show()


# In[6]:


# Here df is our RDD:
df.head()


# ## Now here we gonna print our data schema. 
# ### What is database schema ?
# ### A database schema is the skeleton structure that represents the logical view of the entire database. It defines how the data is organized and how the relations among them are associated. It formulates all the constraints that are to be applied on the data.

# In[7]:


df.printSchema()


# ### After a quick view of schema, to see our spark notebook as in a view of python notebook or to analyse the data as per the libararies of pandas, we use:

# In[8]:


df.describe().toPandas()


# In[9]:


import pandas as pd 
pd.DataFrame(df.take(10),columns=df.columns)


# ## EDA - Data analysis and feature engineering 

# In[10]:


# IMPORT LIBRARIES
from pyspark.sql import *
import matplotlib
import matplotlib.pyplot as plt
from numpy import array
import numpy as np
import numpy as np
import datetime
from pyspark.sql.functions import *
from pyspark import SparkContext


# In[11]:


num_feat = [i[0] for i in df.dtypes if i[1]=='int']


# In[12]:


num_feat = [i[0] for i in df.dtypes if i[1]=='int']
num_feat


# #### Summary of all the numeric attributes

# In[13]:


df.select(num_feat).describe().toPandas()


# In[14]:


df_num = df.select(num_feat).toPandas()


# In[15]:


df_num


# ### To take a instant view of all the available attributes of our dataset we will create Scatetr Matrix, which is quite similar with our PaiPlot by MatplotLib librabries. It shows realtionship between each and every variable and their frequency distribution.

# In[16]:


axs = pd.plotting.scatter_matrix(df_num, figsize = (20, 20))
n = len(df_num.columns)


# In[17]:


histogram = df.select('DayMins').toPandas


# In[ ]:


pip install handyspark


# In[18]:


pd.plotting.hist_frame(df)


# In[ ]:


from handyspark import * # tried to import handy spark to plot histogram but still didn't work.


# ### To see the Spread of all numeric attributes of data, we will gonna create histogram and will see what is the distribution of data on a given range.

# In[73]:


# Histograms for the given numeric columns:
axs = pd.plotting.hist_frame(df_num,figsize=(20,30))
n = len(df_num.columns)


# ### To know the dependency of variable, the best method is to take a look into Correlation between them. Here we'll be creating correlation matrix to compare our numeric variables and to see their relationship.

# In[80]:


df_num.corr()


# ### As per the correlation we can say that there are some variable like CustServCalls and NighCalss which are negatively and postively correlated to the Churn which is our dependent variable. shows that we need to do preprocessing.

# ## Now, after EDA, we need to convert our data so that it will be compatible for our model building process, so for all our categorical variable we use OneHotEncoder, String Indexer and Vector Assembler:

# In[19]:


cat_col = [i[0] for i in df.dtypes if i[1]=='string']
cols = df.columns


# In[20]:


from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
stage = []


# In[21]:


## StringIndexer and Vector Assembler:
for i in cat_col:
    stridx = StringIndexer(inputCol=i, outputCol=i+'_index')
    encoder = OneHotEncoderEstimator(inputCols=[stridx.getOutputCol()], outputCols=[i+'enc'])
    stage += [stridx, encoder]
stage


# In[22]:


label_stringIdx = StringIndexer(inputCol='Churn', outputCol='label')
stage += [label_stringIdx]
df_num_feat = ['AccountLength', 'VMailMessage', 'DayMins', 'EveMins', 'NightMins', 'IntlMins', 'CustServCalls', 'IntlPlan',
               'VMailPlan', 'DayCalls', 'DayCharge', 'EveCalls', 'EveCharge', 'NightCalls', 'NightCharge', 'IntlCalls',
               'IntlCharge', 'AreaCode']


# In[23]:


## Vector Assembler:
assembler_Inputs = [c+'enc' for c in cat_col] + df_num_feat
assembler = VectorAssembler(inputCols=assembler_Inputs, outputCol='features')
stage += [assembler]
stage


# In[24]:


from pyspark.ml import Pipeline
pln = Pipeline(stages=stage)
pln_model = pln.fit(df)
df = pln_model.transform(df)
selected_col = ['label', 'features']+cols
df = df.select(selected_col)


# ## And now, we are splitting our train & test dataset into 75% & 25%:

# In[25]:


train, test = df.randomSplit([0.75, 0.25])
print(train.count())
print(test.count())


# # Model Building 

# ### To build a Machine learning model, we need to start with Model Selection Process:
# ### Then, import all the selected model Libraries
# ### To Evaluate your model, import Evaluator
# ### and To imporvise your model accuracy import all the parameter for ML Tuning
# ## here you gonna use:
# ### 1. Logistic Regression
# ### 2. Desicion Tree
# ### 3. Random Forest

# # 1. Logistic Regression;

# In[ ]:


from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

#Importing 1st LR model and fiting this model plus checking row prediction and probability and churn

lr = LogisticRegression(featuresCol='features', labelCol='label')
lr_model = lr.fit(train)
lr_preds = lr_model.transform(test)
lr_preds.select('rawPrediction', 'probability', 'prediction', 'Churn').show()


# In[26]:


from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

#Importing 1st LR model and fiting this model plus checking row prediction and probability and churn

lr = LogisticRegression(featuresCol='features', labelCol='label')
lr_model = lr.fit(train)
lr_preds = lr_model.transform(test)
lr_preds.select('rawPrediction', 'probability', 'prediction', 'Churn').show()


# ## Evaluation Matrix - PCA Principle Component Analysis;

# In[27]:


# Principle component Analysis:
evaluator = BinaryClassificationEvaluator()
evaluator.evaluate(lr_preds)
training_summary = lr_model.summary
roc = training_summary.roc.toPandas()
roc


# In[28]:


# Ploting ROC Curve:

plt.plot(roc['FPR'],roc['TPR'])
plt.ylabel('True positive Rate')
plt.xlabel('False positive Rate')
plt.title('ROC curve')
plt.show()


# # Grid Search:
# ### What is Grid Search ?
# ### Grid-searching is the process of scanning the data to configure optimal parameters for a given model. Depending on the type of model utilized, certain parameters are necessary. Grid-searching does NOT only apply to one model type. Grid-searching can be applied across machine learning to calculate the best parameters to use for any given model. It is important to note that Grid-searching can be extremely computationally expensive and may take your machine quite a long time to run. Grid-Search will build a model on each parameter combination possible. It iterates through every parameter combination and stores a model for each combination.

# In[30]:


# Building Paramter grid and Cross Validation:

lr_param_grid = (ParamGridBuilder()
              .addGrid(lr.regParam, [0.01, 0.1, 0.5, 1.0, 2.0])
              .addGrid(lr.maxIter, [1, 5, 10, 20, 50]).build())
cv = CrossValidator(estimator = lr,
                    estimatorParamMaps = lr_param_grid,
                    evaluator=evaluator, numFolds=10)
cv_model = cv.fit(train)
cv_lr_preds = cv_model.transform(test)


# In[32]:


cv_lr_preds.select('rawPrediction', 'probability', 'prediction', 'Churn').show()


# In[33]:


preds = lr_model.transform(test)


# In[34]:


pred_summary = preds.select('rawPrediction', 'probability', 'prediction', 'Churn')


# In[35]:


pred_summary.show()


# ### Model Accuracy:

# In[36]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator
cl_eval = BinaryClassificationEvaluator()
cl_eval.evaluate(preds)


# # Decision Tree Model:

# In[37]:


from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
dt_clas = DecisionTreeClassifier(featuresCol='features', labelCol='label', maxDepth=30)


# In[38]:


dt_model = dt_clas.fit(train)


# In[39]:


dt_preds = dt_model.transform(test)


# In[41]:


dt_preds.select('rawPrediction', 'probability', 'prediction', 'Churn').show()


# ### Model accuracy 

# In[42]:


cl_eval.evaluate(dt_preds)


# # Random Forest Model

# In[43]:


rf_clas = RandomForestClassifier(featuresCol='features', labelCol='label', maxDepth=20)


# In[44]:


rf_model = rf_clas.fit(train)


# In[46]:


rf_preds = rf_model.transform(test)


# In[47]:


rf_preds.select('rawPrediction', 'probability', 'prediction', 'Churn').show()


# ### Model Accuracy:

# In[48]:


cl_eval.evaluate(rf_preds)


# ### Random forest model Give Us accuracy of 84.50% which quite well as compare to Logistic regression 

# In[45]:


## Import gridsearch crossvalidator and performing hyperparameter tunning
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


# In[50]:


param_grid_lr = (ParamGridBuilder()
              .addGrid(lr.regParam, [0.01, 0.5, 2.0])
              .addGrid(lr.maxIter, [1, 5, 10]).build())


# In[51]:


cv = CrossValidator(estimator = lr, 
                    estimatorParamMaps = param_grid_lr, 
                    evaluator=cl_eval, numFolds=10)


# In[52]:


cv_model = cv.fit(train)


# In[53]:


cv_lr_preds = cv_model.transform(test)


# In[54]:


cv_lr_preds.select('rawPrediction', 'probability', 'prediction', 'Churn').show()


# In[55]:


cl_eval.evaluate(cv_lr_preds)


# ## After Grid Search we get the accuracy of LR model is 76.24%

# In[56]:


## Performing grid search on Decision Tree Model:
param_grid_dt = (ParamGridBuilder()
              .addGrid(dt_clas.maxDepth, [1, 2, 5, 10, 20])
              .addGrid(dt_clas.maxBins, [20, 32, 40, 80]).build())


# In[57]:


cv_dt = CrossValidator(estimator=dt_clas, 
                       estimatorParamMaps=param_grid_dt, 
                      evaluator=cl_eval, numFolds=5)


# In[58]:


cv_dt_model = cv_dt.fit(train)


# In[59]:


cv_dt_preds = cv_dt_model.transform(test)


# In[60]:


cv_dt_preds.select('rawPrediction', 'probability', 'prediction', 'Churn').show()


# In[61]:


cl_eval.evaluate(cv_dt_preds)


# ### After Grid Search we get the accuracy of DT model is 72.77%

# param_grid_rf = (ParamGridBuilder() .addGrid(rf_clas.maxDepth, [2, 5, 10, 20, 30]) 
#               .addGrid(rf_clas.maxBins, [20, 32, 40, 80]) 
#               .addGrid(rf_clas.numTrees, [5, 10, 20, 40]).build())

# cv_rf = CrossValidator(estimator=rf_clas, estimatorParamMaps=param_grid_rf, evaluator=cl_eval, numFolds=5)

# cv_rf_model = cv_rf.fit(train)

# cv_rf_preds = cv_rf_model.transform(test)

# cv_rf_preds.select('rawPrediction', 'probability', 'prediction', 'Churn').show()

# cl_eval.evaluate(cv_rf_preds)

# In[65]:


# Identifying best model 
best_model = cv_model.bestModel


# In[66]:


predictions = best_model.transform(df)


# In[67]:


cl_eval.evaluate(predictions)


# In[68]:


predictions.createOrReplaceTempView('predictions')


# In[69]:


predictions.show()


# In[70]:


predictions.printSchema()


# In[72]:


spark_2.sql('SELECT * FROM predictions').toPandas()


# ## Conclusion:
# ### As we have build our model on Logistic regression, Decisio tree and Random Forest, we've seen that as compare to others random forest gives us 84.50% accuracy and even after grid search & cross validation on Logistic and Decision tree it gives us somewhere around 74% accuraccy, due to some technical glintch we were not be able to run grid serarch on Random forest, even after that it gives us 84.54% accuracy. shows that its the best fit model for the given dataset and at the same time it leads to our conclusion that our model abble to predict the sentiment of 85% of customers churn on telecom services.

# # Thank You

# In[ ]:




