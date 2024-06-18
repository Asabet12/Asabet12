#!/usr/bin/env python
# coding: utf-8

# ## A2 l Regression Model Development
# ### DAT-5390 | Computational Data Analytics with Python
# #### Hult International Business School
# Ahmed Sabet 
# #### Bibliography
# Hallowell, R., & Danskin, E. (2014). Upselling, cross-selling, and the bottom line. Harvard Business Review, 92(11), 76-84. Retrieved on February 20, 2023.
# 
# Taylor, S. L. S., & Amirloo, P. (2016). Why we prefer products with low prices. Psychology & Marketing, 33(12), 1005-1016. doi: 10.1002/mar.20954. Retrieved on February 20, 2023.
# 
# "Why We Prefer Products With Low Prices" by Steve L. S. Taylor and Ponthea Amirloo, in Psychology & Marketing, Volume 33, Issue 12, Pages 1005-1016 (December 2016). Retrieved on February 20, 2023.
# 
# Stack Overflow. (2020). How to create bins in Python? Retrieved February 20, 2023, from https://stackoverflow.com/questions/63046209/how-to-create-bins-in-python
# 
# Kusterer, C. (2019). Python for Business Analytics [Review of Python for Business Analytics]. Chase Kusterer.
# 
# Stack Overflow. (2016, November 2). Python: Start and End functions - select tuple items [Web log post]. Retrieved February 20, 2023, from https://stackoverflow.com/questions/40362479/python-start-and-end-functions-select-tuple-items
# 
# 
# 

# <h2>Part I: Missing Value Analysis and Imputation</h2><br>
# Importing libraries and the data.

# In[119]:


#Importing Libraries
import pandas as pd                   
import matplotlib.pyplot as plt       
import seaborn as sns                 
import numpy as np                    
import statsmodels.formula.api as smf 
from sklearn.model_selection import train_test_split
import sklearn.linear_model


# setting pandas print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# specifying file name
file = r'C:\Users\Asabe\Downloads\Apprentice_Chef_Dataset_2023.xlsx'


# reading the file into Python
apprentice_chef = pd.read_excel(file)

# Investigating the null values of the apprentice chef dataset
# looking at what features do we have
# and then
# summing together the results per column
apprentice_chef.isnull().sum(axis = 0)

#Analyzing the missing Values and the potential replacements
apprentice_chef[ apprentice_chef.loc[ : , "FAMILY_NAME" ].isnull()].head( n = 10)

#Replacing the Missing Values with the name in Parenthesis
#Could be a Data Entry Mistake

#Extracting Family name from the Name Column
#Stackoverflow (2020)
def extract_family_name(full_name): 
    start = full_name.find("(")
    end = full_name.find(")")
    if start != -1 and end != -1:
        return full_name[start+1:end]
    else:
        return None

# create a new column with the extracted family name
apprentice_chef['FAMILY_NAME_EXTRACTED'] = apprentice_chef['NAME'].apply(extract_family_name)

# replace missing values in the FAMILY_NAME column with the extracted family name
apprentice_chef['FAMILY_NAME'] = np.where(apprentice_chef['FAMILY_NAME'].isnull(), apprentice_chef['FAMILY_NAME_EXTRACTED'], apprentice_chef['FAMILY_NAME'])

# drop the FAMILY_NAME_EXTRACTED column
apprentice_chef = apprentice_chef.drop('FAMILY_NAME_EXTRACTED', axis=1)

#Validating that the Data has been Replaced Successfully
apprentice_chef.iloc[ 54 , :]


# In[120]:


# creating a new column called 'LATE DELIVERIES' since the original has space in the title of the column
apprentice_chef['LATE_DELIVERIES'] = apprentice_chef.iloc[:, 14]

# delete original 'LATE_DELIVERIES ' column with space
apprentice_chef = apprentice_chef.drop(columns="LATE_DELIVERIES ")


# ## <hr style="height:.9px;border:none;color:#333;background-color:#333;" /><br>
# <h3> Classifying the Dataset </h3><br>

# ##### Breaking down the types of data in Database. The process will help in choosing the apporiarte data and group them accordingly.
# Classifying the Dataset

CONTINUOUS
-----------
REVENUE
AVG_TIME_PER_SITE_VISIT
AVG_PREP_VID_TIME
TOTAL_MEALS_ORDERED
--------------
INTERVAL/COUNT
--------------
UNIQUE_MEALS_PURCH
PRODUCT_CATEGORIES_VIEWED
CANCELLATIONS_AFTER_NOON
PC_LOGINS
MOBILE_LOGINS
LATE_DELIVERIES
LARGEST_ORDER_SIZE
AVG_MEAN_RATING
TOTAL_PHOTOS_VIEWED
--------------
CATEGORICAL/OTHER
--------------
NAME
EMAIL
FIRST_NAME
FAMILY_NAME
WEEKLY_PLAN

# In[121]:


#Importing Libraries
import pandas as pd                   
import matplotlib.pyplot as plt       
import seaborn as sns                 
import numpy as np                    
import statsmodels.formula.api as smf 


# setting pandas print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# specifying file name
file = r'C:\Users\Asabe\Downloads\Apprentice_Chef_Dataset_2023.xlsx'


# reading the file into Python
apprentice_chef = pd.read_excel(file)

# Investigating the null values of the apprentice chef dataset
# looking at what features do we have
# and then
# summing together the results per column
apprentice_chef.isnull().sum(axis = 0)

#Analyzing the missing Values and the potential replacements
apprentice_chef[ apprentice_chef.loc[ : , "FAMILY_NAME" ].isnull()].head( n = 10)

#Replacing the Missing Values with the name in Parenthesis
#Could be a Data Entry Mistake

#Extracting Family name from the Name Column
def extract_family_name(full_name): 
    start = full_name.find("(")
    end = full_name.find(")")
    if start != -1 and end != -1:
        return full_name[start+1:end]
    else:
        return None

# create a new column with the extracted family name
apprentice_chef['FAMILY_NAME_EXTRACTED'] = apprentice_chef['NAME'].apply(extract_family_name)

# replace missing values in the FAMILY_NAME column with the extracted family name
apprentice_chef['FAMILY_NAME'] = np.where(apprentice_chef['FAMILY_NAME'].isnull(), apprentice_chef['FAMILY_NAME_EXTRACTED'], apprentice_chef['FAMILY_NAME'])

# drop the FAMILY_NAME_EXTRACTED column
apprentice_chef = apprentice_chef.drop('FAMILY_NAME_EXTRACTED', axis=1)

#Validating that the Data has been Replaced Successfully
apprentice_chef.iloc[ 54 , :]


# In[122]:


#Developing a histogram using HISTPLOT
apprentice_chef['log_REVENUE'] = np.log(apprentice_chef['REVENUE'])

sns.histplot(data   = apprentice_chef,
             x      = 'log_REVENUE',
             kde    = True)


# title and axis labels
plt.title(label   = "Logarithmic Distribution of REVENUE")
plt.xlabel(xlabel = "REVENUE") # avoiding using dataset labels
plt.ylabel(ylabel = "Count")

# displaying the histogram
plt.show()


# In[123]:


# creating a (Pearson) correlation matrix
#High level view for which features are 
#Strong Correlated with Revenue
df_corr = apprentice_chef.corr().round(2)


# printing (Pearson) correlations with SalePrice
df_corr.loc[ : , ['REVENUE', 'log_REVENUE'] ].sort_values(by = 'REVENUE',
                                                                ascending = False)


# In[124]:



# displaying the plot for 'AVG_Time_Per_Vist
#Data seems to be skewed to the right
#Using the Log to Normalize the data 
sns.histplot(x = 'AVG_TIME_PER_SITE_VISIT',
             data = apprentice_chef,
             kde = True)


# title and labels
plt.title('AVG_TIME_PER_SITE_VISIT')
plt.xlabel('AVG_TIME_PER_SITE_VISIT')
plt.ylabel('Frequency')

# displaying the plot
plt.show()


# In[125]:


#Developing a histogram using HISTPLOT
apprentice_chef['log_AVG_TIME_PER_SITE_VISIT'] = np.log(apprentice_chef['AVG_TIME_PER_SITE_VISIT'])

sns.histplot(data   = apprentice_chef,
             x      = 'log_AVG_TIME_PER_SITE_VISIT',
             kde    = True)


# title and axis labels
plt.title(label   = 'log_AVG_TIME_PER_SITE_VISIT')
plt.xlabel(xlabel = "AVG_TIME_PER_SITE_VISIT") # avoiding using dataset labels
plt.ylabel(ylabel = "Count")

# displaying the histogram
plt.show()


# In[126]:


# displaying the plot for AVG_PREP_VID_TIME
#Data seems to be skewed to the right
sns.histplot(x = 'CONTACTS_W_CUSTOMER_SERVICE',
             data = apprentice_chef,
             kde = True)


# title and labels
plt.title('AVG_PREP_VID_TIME Distribution')
plt.xlabel('AVG_PREP_VID_TIME')
plt.ylabel('Frequency')

# displaying the plot
plt.show()


# In[127]:


#Developing a histogram using HISTPLOT
apprentice_chef['log_CONTACTS_W_CUSTOMER_SERVICE'] = np.log(apprentice_chef['CONTACTS_W_CUSTOMER_SERVICE'])

sns.histplot(data   = apprentice_chef,
             x      = 'log_CONTACTS_W_CUSTOMER_SERVICE',
             kde    = True)


# title and axis labels
plt.title(label   = 'log_CONTACTS_W_CUSTOMER_SERVICE')
plt.xlabel(xlabel = 'log_TOTAL_PHOTOS_VIEWED') # avoiding using dataset labels
plt.ylabel(ylabel = "Count")

# displaying the histogram
plt.show()


# In[128]:


#Developing a histogram using HISTPLOT
apprentice_chef['log_TOTAL_PHOTOS_VIEWED'] = np.log(apprentice_chef['TOTAL_PHOTOS_VIEWED'])

sns.histplot(data   = apprentice_chef,
             x      = 'log_TOTAL_PHOTOS_VIEWED',
             kde    = True)


# title and axis labels
plt.title(label   = 'log_TOTAL_PHOTOS_VIEWED')
plt.xlabel(xlabel = 'log_TOTAL_PHOTOS_VIEWED') # avoiding using dataset labels
plt.ylabel(ylabel = "Count")

# displaying the histogram
plt.show()


# In[129]:


#Developing a histogram using HISTPLOT
apprentice_chef['log_AVG_PREP_VID_TIME'] = np.log(apprentice_chef['AVG_PREP_VID_TIME'])

sns.histplot(data   = apprentice_chef,
             x      = 'log_AVG_PREP_VID_TIME',
             kde    = True)


# title and axis labels
plt.title(label   = 'log_AVG_TIME_PER_SITE_VISIT')
plt.xlabel(xlabel = 'log_AVG_PREP_VID_TIME') # avoiding using dataset labels
plt.ylabel(ylabel = "Count")

# displaying the histogram
plt.show()


# In[130]:


#Developing a histogram using HISTPLOT
apprentice_chef['log_AVG_MEAN_RATING'] = np.log(apprentice_chef['AVG_MEAN_RATING'])

sns.histplot(data   = apprentice_chef,
             x      = 'log_AVG_MEAN_RATING',
             kde    = True)


# title and axis labels
plt.title(label   = 'log_AVG_MEAN_RATING')
plt.xlabel(xlabel = 'log_AVG_PREP_VID_TIME') # avoiding using dataset labels
plt.ylabel(ylabel = "Count")

# displaying the histogram
plt.show()


# In[131]:


#Developing a histogram using HISTPLOT
apprentice_chef['log_TOTAL_MEALS_ORDERED'] = np.log(apprentice_chef['TOTAL_MEALS_ORDERED'])

sns.histplot(data   = apprentice_chef,
             x      = 'log_AVG_PREP_VID_TIME',
             kde    = True)


# title and axis labels
plt.title(label   = 'log_TOTAL_MEALS_ORDERED')
plt.xlabel(xlabel = 'log_AVG_PREP_VID_TIME') # avoiding using dataset labels
plt.ylabel(ylabel = "Count")

# displaying the histogram
plt.show()


# <h2> Categories Viewed Analysis Developing Trend-Based Features</h2><br>
# <span style="color:red"><h3> Product Categories Viewed </h3><br></span>
# <span style="color:red"><h4> Mean 5 and Revenue @ 9 </h4><br></span>

#  A scatter plot that illustrates some interesting findings is shown below. It seems that when a customer sees five different products, that is when the greatest amount of revenue is achieved. This might be because when the customer sees too many different products, they become overwhelmed with information and do not end up making a purchase. This can be referred to as decision paralysis where the customer avoids making a decision altogether. (Iyengar, S. S., & Lepper, M. R. (2000)

# In[132]:


#Compare Same Feature vs Log_Revenue this time instead of Revenue
# setting figure size
fig, ax = plt.subplots(figsize = (4, 4))

sns.scatterplot(x = apprentice_chef['PRODUCT_CATEGORIES_VIEWED'],
                y = apprentice_chef['log_REVENUE'],
                color = 'r')


# In[133]:


# setting figure size
fig, ax = plt.subplots(figsize = (4, 4))


# developing a scatterplot for Number of Product Categories Viewed
sns.scatterplot(x = apprentice_chef['PRODUCT_CATEGORIES_VIEWED'],
                y = apprentice_chef['REVENUE'],
                color = 'r')


# ## Cancellations  & Customer interactions Service Analysis 
# ### Developing Trend-Based Features 
# #### <span style="color:red"> Order Fullfilment 

# Another interesting finding is that the average revenue is at its highest at the mean point. What caught my attention is the fact that revenue increases with more canceled orders. However, if we look at the cancellation policy, it makes sense since customers only receive a partial refund if it's after noon. Also, one might ask why it does not increase as the number increases, which is a logical assumption. That's why we have conducted value counts below, and we can see that we have more data samples from 1 to 5 observations, and as our cancellation count increases, our samples decrease. In this case, I can conclude that the data tends to favor those observations, and we do not have enough data to come to a logical conclusion about which cancellation count would generate the most revenue.

# <span style="color:red"><h3>Customer Service Interaction</h3></span>

# In[134]:


#Relationship of Cancellation orders and Revneue
# setting figure size
fig, ax = plt.subplots(figsize = (15, 10))


# developing a boxplot
plt.subplot(1, 2, 1)
sns.boxplot(x     = 'CANCELLATIONS_AFTER_NOON',
            y     = 'log_REVENUE',
            data  = apprentice_chef)


# Title and Labels
plt.title(label   = 'Relationship between CANCELLATIONS_AFTER_NOON vs REVENUE')
plt.xlabel(xlabel = 'CANCELLATIONS_AFTER_NOON')
plt.ylabel(ylabel = 'log_REVENUE')


# In[135]:



#Relationship of Cancellation orders and Revneue
# setting figure size
fig, ax = plt.subplots(figsize = (15, 10))


# developing a boxplot
plt.subplot(1, 2, 1)
sns.boxplot(x     = 'CONTACTS_W_CUSTOMER_SERVICE',
            y     = 'log_REVENUE',
            data  = apprentice_chef)


# titles and labels
plt.title(label   = 'Relationship between Contacts W/Customer Service')
plt.xlabel(xlabel = 'Contacts W/Customer Service')
plt.ylabel(ylabel = 'log_REVENue')


# In[136]:


# printing value counts for full and half baths
print(f"""
CANCELLATIONS AFTER NOON
==================

----------
CANCELLATIONS AFTER NOON
----------
{apprentice_chef['CANCELLATIONS_AFTER_NOON'].value_counts(normalize = False).sort_index()}

----------
""")


# In[137]:


# printing value counts for full and half baths
print(f"""
CONTACTS W/CUSTOMER SERVICE
==================

CONTACTS W/CUSTOMER SERVICE
----------
{apprentice_chef['CONTACTS_W_CUSTOMER_SERVICE'].value_counts(normalize = False).sort_index()}


""")


# <h2> Contacts With Customer Service Developing Trend-Based Features</h2><br>
# <span style="color:red"><h3> Complaint Rate </h3><br></span>
# This is a feature designed to understand the complaint rate. It takes several factors into account: contacts with customer service, cancellations after noon, PC logins, and mobile logins. It assumes that the PC logins and mobile logins were to contact customer service. The featured company performed as expected: the higher the ratio, the greater the inverse relationship between revenue and ratio. 

# In[138]:


#Creating a New Feature based on Customer Interactions
#Combining the PC_Logins, Mobile_Logins 
#Cancellations after noon, Customer Service
#Instantiating the Ratio
apprentice_chef['High_Customer_Complaint_Rate'] = (  apprentice_chef['CONTACTS_W_CUSTOMER_SERVICE']  + apprentice_chef['CANCELLATIONS_AFTER_NOON'] + apprentice_chef['PC_LOGINS'] + apprentice_chef['MOBILE_LOGINS']  ) / apprentice_chef['TOTAL_MEALS_ORDERED']

# developing a correlation matrix
df_corr = apprentice_chef.corr().round(2)
# printing (Pearson) correlations with SalePrice
Customer_Complaint_Corr = df_corr.loc[ 'High_Customer_Complaint_Rate' , ['REVENUE', 'log_REVENUE'] ]

# printing correlations of the new created Feature
print(f"""

---------------------
Customer Complaint Corr
---------------------
{Customer_Complaint_Corr}

""")


# In[139]:


#Creating a New Feature based on Customer Interactions
#Combining the PC_Logins, Mobile_Logins 
#Cancellations after noon, Customer Service
#Instantiating the Ratio
apprentice_chef['High_Value_Customer'] = (  apprentice_chef['UNIQUE_MEALS_PURCH']  + apprentice_chef['PC_LOGINS'] + apprentice_chef['PRODUCT_CATEGORIES_VIEWED'] + apprentice_chef['MOBILE_LOGINS']  ) / apprentice_chef['TOTAL_MEALS_ORDERED']

# developing a correlation matrix
df_corr = apprentice_chef.corr().round(2)
# printing (Pearson) correlations with SalePrice
High_Value_Customer_Corr = df_corr.loc[ 'High_Value_Customer' , ['REVENUE', 'log_REVENUE'] ]

# printing correlations of the new created Feature
print(f"""

---------------------
Customer Complaint Corr
---------------------
{High_Value_Customer_Corr}

""")


# <span style="color:red"><h3> Complaint Rate </h3><br></span>
# The graph below reaffirms our assumption above. However, we notice some interesting findings in the graph: some users have high complaint rates yet maintain the same level of revenue as other users with low complaint rates. This could be due to several reasons: lack of substitute products in the market, cost-benefit analysis It will cost the user more if he were to unsubscribe from the meal plan and opt for takeout.

# In[140]:


#Relationship of Cancellation orders and Revneue
# setting figure size
fig, ax = plt.subplots(figsize = (15, 10))


# developing a boxplot
plt.subplot(1, 2, 1)
sns.scatterplot(x     = 'High_Customer_Complaint_Rate',
            y     = 'log_REVENUE',
            data  = apprentice_chef)


# Title and Labels
plt.title(label   = 'Relationship between Complaint Rate and Revenue')
plt.xlabel(xlabel = 'Complaint Rate')
plt.ylabel(ylabel = 'log_REVENUE')

plt.show()


# <h2> Total Meals Ordered Developing Trend-Based Features</h2><br>
# 
# Conducting descriptive statistics to understand the distribution of our data. We can see from the descriptive statistics that 25% of our customers had an average order of 39 meals per year, 50% of the database was at 60 meals, 75% at 95, and finally the last 25% at 493. We need to look into the last two tiers further because they appear to be corporate domains rather than individual domains, with a very large order size compared to the rest of the database. 

# In[141]:


#looking at the Frequency of the of the orders
# Determing the where the data is located
#Objetctive is to construct interval data tier classificication
#This will be based on the Four Tiers below
apprentice_chef['TOTAL_MEALS_ORDERED'].describe().round(decimals = 1)


# <h2> Order Ferquency Cohort Tiers </h2><br>
# <span style="color:red"><h3> Order Count Tiers </h3><br></span>
# A very interesting finding that tells us about our customers' five different cohort levels. It seems that we have a lot of orders going to domain groups or corporate groups. This is based on the assumption that the larger the meal order, the more likely it is for a corporate company where more than one individual is ordering. Individuals, on the other hand, have placed numerous meal orders as well . My actionable insight would be to target these domain groups, as they seem to affect the individual level as well. People might still be ordering seperately. Further, partnerships with corporate companies, SMEs, and SOHO can lead to increased volume sales, enhance brand positioning, and even open opportunities for a new target segment. (Hou and Hartman, 2019)

# In[142]:


#Creating a code to classify the data into four different groups
# The four cohorts are based on the descriptive statistics above
#stackoverflow
apprentice_chef['cohorts'] = pd.cut(apprentice_chef['TOTAL_MEALS_ORDERED'] , 
bins=[0, 11, 39, 60, 95, np.inf], 
labels=['0-11', '11-39', '39-60', '60-95', '>95'])

# Counting customers in each cohort
cohorts_counts = apprentice_chef['cohorts'].value_counts(sort = True,
                                                       ascending = True)
print(cohorts_counts)


# <span style="color:red"><h3> Order Count Tiers </h3><br></span>
# <span style="color:red"><h4> 39-60 cohort @ 7.3 USD & 60-95 @ 7.5 USD  Tiers </h4><br></span>
# <span style="color:red"><h5> Currency in (Thousands) USD   </h5><br></span>
# 
# The graph indeed has confirmed our understanding. However, another important factor to look is that the average revenue for
# 39-60 cohort and 60-95 are very close. This is because the 60-95 group might be opting for lower priced meals. I would suggest to the marketing team to try upsell more expensive products to generate more revenue from this group. Also, Upselling increases revenue, improves customer satisfaction, and enhances brand loyalty.(Hallowell, R., & Danskin, E. (2014)

# In[143]:


# Boxplot - Cohorts and Log_Revenue
sns.boxplot(x    = 'cohorts',
                y    = 'log_REVENUE',
                data = apprentice_chef)


# titles and axis labels
plt.title(label   = 'Boxplot with Cohorts')
plt.xlabel(xlabel = 'Meal Order Tiers')
plt.ylabel(ylabel = 'REVENUE')


# displaying the plot
plt.show()


# <h2> E-mail Cohort Tiers </h2><br>
# <span style="color:red"><h3> E-mail cohort  Tiers </h3><br></span>
# To reaffirm, our assumption above I decided to break down the customer type by domain. We find some interesting findings. Personal e-mail makes up most of our customer base however we might find the same customer in the domain group as well but ordered by a different individual. I would focus on increasing our domain group as this can lead to multiple effects and create a larger customer base.

# In[144]:


def classify_emails(email):
    if email.endswith(('@mmm.com', '@amex.com', '@apple.com', '@boeing.com', '@caterpillar.com', '@chevron.com', '@cisco.com', '@cocacola.com', '@disney.com', '@dupont.com', '@exxon.com', '@ge.org', '@goldmansacs.com', '@homedepot.com', '@ibm.com', '@intel.com', '@jnj.com', '@jpmorgan.com', '@mcdonalds.com', '@merck.com', '@microsoft.com', '@nike.com', '@pfizer.com', '@pg.com', '@travelers.com', '@unitedtech.com', '@unitedhealth.com', '@verizon.com', '@visa.com', '@walmart.com')):
        return 0
    elif email.endswith(('@gmail.com', '@yahoo.com', '@protonmail.com')):
        return 1
    elif email.endswith(('@me.com', '@aol.com', '@hotmail.com', '@live.com', '@msn.com', '@passport.com')):
        return 2
    else:
        return -1

for index, row in apprentice_chef.iterrows():
    email = row['EMAIL']
    apprentice_chef.loc[index, 'CUSTOMER_SEGMENT'] = classify_emails(email)


# In[145]:


apprentice_chef['CUSTOMER_SEGMENT'].value_counts (normalize = True,
                                                    ascending =  True)


# In[146]:


#Developing COrrelation Matrix by Customer Segment
df_corr = apprentice_chef.corr().round(2)
# printing (Pearson) correlations with SalePrice
CUSTOMER_SEGEMENT_CORR = df_corr.loc[ 'CUSTOMER_SEGMENT' , ['REVENUE', 'log_REVENUE'] ]

# printing correlations of the new created Feature
print(f"""

---------------------
CUSTOMER SEGMENT CORR
---------------------
{CUSTOMER_SEGEMENT_CORR}

""")


# <h2> Feature Engineering Features </h2><br>
# Feature Enginnering some new features to enahnce model accuracy. Not all the feataures were integrated to the model other proved to be vital to the model's succes. Some interesting features that were developed were User interaction, Weekly Plan optins and the avg time per site vs the total orders ordered

# In[148]:


df_corr = apprentice_chef.corr().round(2)
# printing (Pearson) correlations with SalePrice
USER_INTERACTION_CORR = df_corr.loc[ 'USER_INTERACTION', ['REVENUE', 'log_REVENUE'] ]

# printing correlations of the new created Feature
print(f"""

---------------------
USER INTERACTION CORR
---------------------
{USER_INTERACTION_CORR}

""")


# ### Developing Trend-Based Features 
# #### <span style="color:red"> Customer Orders
#     
# Another Features was created which is customer orders. The feature is made up of the total unique purhcase plus the total photos viewed divied by the total meals ordered. This might make sense to some extent. This probably that this users are looking low qty and speicific items and do not mass order. As we can see from our correlation numbers there exits an inverse relationsip between revenue and customer orders.

# In[149]:


f_corr = apprentice_chef.corr().round(2)
# printing (Pearson) correlations with SalePrice
Custom_Orders_CORR = df_corr.loc[ 'Custom_Orders', ['REVENUE', 'log_REVENUE'] ]

# printing correlations of the new created Feature
print(f"""

---------------------
CUSTOM ORDERS
---------------------
{Custom_Orders_CORR}

""")


# #### <span style="color:red"> Customer Orders
# 
# In the Scatterplot belove it becomes clear that most of the data is in the lower limit of the ratio and has the maximum revenue. I would suggest to the pricing team to have a look at the pricing structure since it seems that Apprentice Chef is not making that much money from customer orders.

# In[150]:


# Boxplot - Cohorts and Log_Revenue
sns.scatterplot(x    = 'Custom_Orders',
                y    = 'log_REVENUE',
                data = apprentice_chef)


# titles and axis labels
plt.title(label   = 'Scatterplot with Custom_Orders')
plt.xlabel(xlabel = 'Custom_Orders')
plt.ylabel(ylabel = 'REVENUE')


# displaying the plot
plt.show()


# In[151]:


#Instantiating a placegolder for Customers
#that rate orders greater than the average
apprentice_chef['High_Prep_Time'] = 0

# Checking for a higher rating than the mean
for index, value in apprentice_chef.iterrows():    
    
    if apprentice_chef.loc[index, 'AVG_PREP_VID_TIME'] > 100:
        apprentice_chef.loc[index,'High_Prep_Time' ] = 1
    
    elif apprentice_chef.loc[index, 'AVG_PREP_VID_TIME'] < 100:
        apprentice_chef.loc[index,'High_Prep_Time' ] = 0


# In[152]:


apprentice_chef['High_Prep_Time'].value_counts(normalize = False,
                                                     sort      = True)


# In[153]:


#Instantiating a placegolder for Customers
#that have Large order size greater than the Average
apprentice_chef['Mega Orders'] = 0

# Checking for a higher rating than the mean
for index, value in apprentice_chef.iterrows():    
    
    if apprentice_chef.loc[index, 'LARGEST_ORDER_SIZE'] > 4:
        apprentice_chef.loc[index,'Mega Orders' ] = 1
    
    elif apprentice_chef.loc[index, 'LARGEST_ORDER_SIZE'] < 4:
        apprentice_chef.loc[index,'Mega Orders' ] = 0


# In[154]:


#Instantiating a placegolder for Customers
#that have Large order size greater than the Average
apprentice_chef['Different Tastes'] = 0

# Checking for a higher rating than the mean
for index, value in apprentice_chef.iterrows():    
    
    if apprentice_chef.loc[index, 'UNIQUE_MEALS_PURCH'] > 7:
        apprentice_chef.loc[index,'Different Tastes' ] = 1
    
    elif apprentice_chef.loc[index, 'UNIQUE_MEALS_PURCH'] < 7:
        apprentice_chef.loc[index,'Different Tastes'  ] = 0


# In[205]:


#Instantiating a placegolder for Customers
#that have Large order size greater than the zobr
apprentice_chef['Meal Searching'] = 0

# Checking for a higher rating than the mean
for index, value in apprentice_chef.iterrows():    
    
    if apprentice_chef.loc[index, 'TOTAL_PHOTOS_VIEWED'] > 113:
        apprentice_chef.loc[index,'Meal Searching' ] = 1
    
    elif apprentice_chef.loc[index, 'TOTAL_PHOTOS_VIEWED'] < 113:
        apprentice_chef.loc[index,'Meal Searching' ] = 0


# In[204]:


df_corr = apprentice_chef.corr().round(2)
High_Customer_Statisfaction = df_corr.loc[ 'High_Customer_Statisfaction', ['REVENUE', 'log_REVENUE'] ]

print(f"""

---------------------
HIGH_CUSTOMER_STATISFACTION
---------------------
{High_Customer_Statisfaction}

""")


# In[177]:


#Instantiating a placegolder for Customers
#that rate orders greater than the average
apprentice_chef['High_Customer_Statisfaction'] = 0

# Checking for a higher rating than the mean
for index, value in apprentice_chef.iterrows():    
    
    if apprentice_chef.loc[index, 'AVG_MEAN_RATING'] > 2.5:
        apprentice_chef.loc[index,'High_Customer_Statisfaction' ] = 1
    
    elif apprentice_chef.loc[index, 'AVG_MEAN_RATING'] < 2.5:
        apprentice_chef.loc[index,'High_Customer_Statisfaction' ] = 0


# In[158]:


#Instantiating a placegolder for Customers
#for order meal levels
#Grouping the data optimal pattern identification
apprentice_chef['Meal Order Tiers'] = 0

# Checking for a higher rating than the mean
for index, value in apprentice_chef.iterrows():    
    
    if apprentice_chef.loc[index, 'TOTAL_MEALS_ORDERED'] <= 100:
        apprentice_chef.loc[index,'Meal Order Tiers' ] = 1
    
    elif apprentice_chef.loc[index, 'AVG_MEAN_RATING'] <= 200:
        apprentice_chef.loc[index,'Meal Order Tiers' ] = 2
    
    elif apprentice_chef.loc[index, 'AVG_MEAN_RATING'] <= 300:
        apprentice_chef.loc[index,'Meal Order Tiers' ] = 3
    
    elif apprentice_chef.loc[index, 'AVG_MEAN_RATING'] <= 400:
        apprentice_chef.loc[index,'Meal Order Tiers' ] = 4
    
    elif apprentice_chef.loc[index, 'AVG_MEAN_RATING'] <= 500:
        apprentice_chef.loc[index,'Meal Order Tiers' ] = 5
              


# In[159]:


#Instantiating a placegolder for Customers
#that rate orders greater than the average
apprentice_chef['Website Activity'] = 0

# Checking for a higher rating than the mean
for index, value in apprentice_chef.iterrows():    
    
    if apprentice_chef.loc[index,'AVG_TIME_PER_SITE_VISIT'] <= 100:
        apprentice_chef.loc[index,'Website Activity' ] = 1
    
    elif apprentice_chef.loc[index,'AVG_TIME_PER_SITE_VISIT'] <= 200:
        apprentice_chef.loc[index,'Website Activity' ] = 2
    
    elif apprentice_chef.loc[index,'AVG_TIME_PER_SITE_VISIT'] <= 300:
        apprentice_chef.loc[index,'Website Activity' ] = 3
    
    elif apprentice_chef.loc[index,'AVG_TIME_PER_SITE_VISIT'] <= 400:
        apprentice_chef.loc[index,'Website Activity' ] = 4
    
    elif apprentice_chef.loc[index,'AVG_TIME_PER_SITE_VISIT'] <= 500:
        apprentice_chef.loc[index,'Website Activity' ] = 5
        
    elif apprentice_chef.loc[index,'AVG_TIME_PER_SITE_VISIT'] <= 600:
        apprentice_chef.loc[index,'Website Activity' ] = 6
              
              


# In[160]:


# Instantiating a placeholder for customers contacts W/Customer Service
apprentice_chef['Customer Morale'] = 0
low_tier = [1, 2, 3, 4, 5]
medium_tier = [6, 7, 8, 9, 10]
high_tier  = [11, 12, 13, 18, 19]

# Checking for a higher rating than the mean
for index, value in apprentice_chef.iterrows():    
    
    if apprentice_chef.loc[index, 'CONTACTS_W_CUSTOMER_SERVICE'] in low_tier:
        apprentice_chef.loc[index, 'Customer Morale'] = 1
    
    elif apprentice_chef.loc[index, 'CONTACTS_W_CUSTOMER_SERVICE'] in medium_tier:
        apprentice_chef.loc[index, 'Customer Morale'] = 2
    
    elif apprentice_chef.loc[index, 'CONTACTS_W_CUSTOMER_SERVICE'] in high_tier:
        apprentice_chef.loc[index, 'Customer Morale'] = 3
              


# In[161]:


apprentice_chef['DIVERSE MEALS TIERS'] = 0
low_tier = [1, 2, 3, 4, 5]
medium_tier = [6, 7, 8, 9, 10]
high_tier  = [11, 12, 13, 18, 19]

for index, value in apprentice_chef.iterrows():    
    unique_meals_purchased = apprentice_chef.loc[index, 'UNIQUE_MEALS_PURCH']
    if unique_meals_purchased in low_tier:
        apprentice_chef.loc[index, 'DIVERSE MEALS TIERS'] = 1
    elif unique_meals_purchased in medium_tier:
        apprentice_chef.loc[index, 'DIVERSE MEALS TIERS'] = 2
    elif unique_meals_purchased in high_tier:
        apprentice_chef.loc[index, 'DIVERSE MEALS TIERS'] = 3


# In[162]:


#Instantiating a placegolder for Customers
#that rate orders greater than the average
apprentice_chef['Reliability'] = 0

# Checking for a higher rating than the mean
for index, value in apprentice_chef.iterrows():    
    
    if apprentice_chef.loc[index, 'CANCELLATIONS_AFTER_NOON'] < 10:
        apprentice_chef.loc[index,'Reliability' ] = 1
    
    elif apprentice_chef.loc[index, 'CANCELLATIONS_AFTER_NOON'] > 10:
        apprentice_chef.loc[index,'Reliability' ] = 0


# In[163]:


#Instantiating a placegolder for Customers
#that rate orders greater than the average
apprentice_chef['High Volume'] = 0

# Checking for a higher rating than the mean
for index, value in apprentice_chef.iterrows():    
    
    if apprentice_chef.loc[index, 'TOTAL_MEALS_ORDERED'] < 50:
        apprentice_chef.loc[index,'High Volume' ] = 1
    
    elif apprentice_chef.loc[index, 'TOTAL_MEALS_ORDERED'] > 50:
        apprentice_chef.loc[index,'High Volume' ] = 0


# ### Customer Statisfaction BoxPlot 
# 
# Interesting Findings we can see that the average log_REVENUE for the Customer Ratings above the mean is lower than the average log_revenue with customers with lower ratings. This means that customers are more satisfied with meals that are lower in price. Therefore, there exists an inverse relationship between Customer Ratings and Revenue. It might be possible that customers are more statisfied when they pay less for a number a reasons becase it is an indication that they are getting value for money in this case. On ther hand, it might because customer paid too much for a meal that they did not end up liking.(Taylor and  Amirloo, 2016)

# In[184]:


sns.boxplot(x    = 'High_Customer_Statisfaction',
            y    = 'log_REVENUE',
            data = apprentice_chef)


# titles and axis labels
plt.title(label   = 'Relationship Between Customer Statisfaction & Revenue')
plt.xlabel(xlabel = 'Customer_Ratings_AboveMEAN')
plt.ylabel(ylabel = 'log_REVENUE')


# displaying the plot
plt.show()


# In[147]:


#Instatitaing a new feature
#that might be valueable for the model
#Combining Feature that would make sense to be grouped together

apprentice_chef['USER_INTERACTION'] = (apprentice_chef['PC_LOGINS'] + apprentice_chef['MOBILE_LOGINS'] )/apprentice_chef['AVG_TIME_PER_SITE_VISIT']
apprentice_chef['Custom_Orders' ]   = (apprentice_chef['UNIQUE_MEALS_PURCH'] + apprentice_chef['PRODUCT_CATEGORIES_VIEWED']) / apprentice_chef['TOTAL_MEALS_ORDERED']
apprentice_chef['Largest order size to Total Orders' ] = apprentice_chef['LARGEST_ORDER_SIZE'] / apprentice_chef['TOTAL_MEALS_ORDERED']
apprentice_chef['Weekly Optins' ] = apprentice_chef['WEEKLY_PLAN']/apprentice_chef['TOTAL_MEALS_ORDERED']
apprentice_chef ['Avg Prep Per Meal'] = apprentice_chef['AVG_PREP_VID_TIME']/apprentice_chef['TOTAL_MEALS_ORDERED']
apprentice_chef ['Avg Time vs Orders'] = apprentice_chef['AVG_TIME_PER_SITE_VISIT']/apprentice_chef['TOTAL_MEALS_ORDERED']
apprentice_chef [ 'Ordered Meals Per Photo' ] = apprentice_chef['TOTAL_MEALS_ORDERED']/apprentice_chef['PRODUCT_CATEGORIES_VIEWED']
apprentice_chef ['Cancellation Frequency']  = apprentice_chef ['CANCELLATIONS_AFTER_NOON']/ apprentice_chef['TOTAL_MEALS_ORDERED']
apprentice_chef ['Categories Frequency']  = apprentice_chef ['PRODUCT_CATEGORIES_VIEWED']/ apprentice_chef['AVG_TIME_PER_SITE_VISIT']
apprentice_chef ['Time to Unique Orders'] = apprentice_chef ['AVG_TIME_PER_SITE_VISIT']/apprentice_chef['UNIQUE_MEALS_PURCH']
apprentice_chef['Largest order size to Total Pohots' ] = apprentice_chef['LARGEST_ORDER_SIZE'] / apprentice_chef['TOTAL_PHOTOS_VIEWED']
apprentice_chef['Largest order size to Avg Time' ] = apprentice_chef['LARGEST_ORDER_SIZE'] / apprentice_chef['AVG_TIME_PER_SITE_VISIT']


# #### Relationship Between Ordered Meals Per Photo & Revenue
# This reaffirms our assumption about when users get overwhelmed with too much information and do not end up ordering. As we can see the highest revenue is achieved for those who viewed the least amount of photos

# In[209]:


sns.scatterplot(x    = 'Ordered Meals Per Photo',
            y    = 'log_REVENUE',
            data = apprentice_chef)


# titles and axis labels
plt.title(label   = 'Relationship Between Ordered Meals Per Photo & Revenue')
plt.xlabel(xlabel = 'Ordered Meals Per Photo')
plt.ylabel(ylabel = 'log_REVENUE')


# displaying the plot
plt.show()


# #### Relationship  Avg Prep Per Meal & Revenue
# The findings below suggest that customers prefer meals that do not take too long to cook, as it might be convenient and time-saving for meal preparation. Further evidence suggests that consumers rate convenience as the most important factor when choosing ready-to-eat meals.

# In[213]:


sns.scatterplot(x    = 'Avg Prep Per Meal',
            y    = 'log_REVENUE',
            data = apprentice_chef)


# titles and axis labels
plt.title(label   = 'Relationship Between Avg Prep Per Meal & Revenue')
plt.xlabel(xlabel = 'Avg Prep per Meal')
plt.ylabel(ylabel = 'log_REVENUE')


# displaying the plot
plt.show()


# #### Relationship  Weekly Optins & Revenue
# 

# Another interesting finding that might be worth investigating is that it seems that customers are not enjoying the weekly subscription feature and do not find it convenient. It would be worthwhile to present the following data to the marketing team and restructure the strategy as it does not seem to be working out.

# In[221]:


sns.scatterplot(x    = 'Weekly Optins',
            y    = 'log_REVENUE',
            data = apprentice_chef)


# titles and axis labels
plt.title(label   = 'Weekly Optins & Revenue')
plt.xlabel(xlabel = 'Weekly Optins')
plt.ylabel(ylabel = 'log_REVENUE')


# displaying the plot
plt.show()


# #### Largest Order Size Total Orders & Revenue
#  Based on the graph below people prefer smaller orders over larger ones. As we can see the lower the ratio the hgiher the revenue. Which means that main driver behind the revenue increase is the small orders. This might because smaller orders can be less overwhelming and more manageable. Some people may feel like they have more control over their meals when they can order in smaller portions. Also, smaller orders can provide greater variety and allow people to try different meals. Instead of committing to one large meal, they can try other ones. In addition, smaller orders can be more affordable, especially for those on a budget. Finally, some people may prefer smaller orders for health reasons. They may want to watch their portion sizes and limit their calorie intake. 

# In[219]:


sns.scatterplot(x    = 'Largest order size to Total Orders',
            y    = 'log_REVENUE',
            data = apprentice_chef)


# titles and axis labels
plt.title(label   = 'Relationship Between Largest order size to Total Orders Meal & Revenue')
plt.xlabel(xlabel = 'Avg Prep per Meal')
plt.ylabel(ylabel = 'log_REVENUE')


# displaying the plot
plt.show()


# In[ ]:


sns.scatterplot(x    = 'Weekly Optins',
            y    = 'log_REVENUE',
            data = apprentice_chef)


# titles and axis labels
plt.title(label   = 'Relationship Between Avg Prep Per Meal & Revenue')
plt.xlabel(xlabel = 'Avg Prep per Meal')
plt.ylabel(ylabel = 'log_REVENUE')


# displaying the plot
plt.show()


# ## Building a Predictive Models

# In[188]:


x_var = [    'log_TOTAL_MEALS_ORDERED',
             'LARGEST_ORDER_SIZE',  'log_TOTAL_PHOTOS_VIEWED', 
             'log_AVG_MEAN_RATING', 'log_CONTACTS_W_CUSTOMER_SERVICE', 
             'UNIQUE_MEALS_PURCH','log_AVG_PREP_VID_TIME', 'log_AVG_TIME_PER_SITE_VISIT', 
             'Largest order size to Total Orders','High_Customer_Statisfaction',
                'Largest order size to Total Pohots' ]

x_var_log_y = [  'log_TOTAL_MEALS_ORDERED',
                 'LARGEST_ORDER_SIZE',  'log_TOTAL_PHOTOS_VIEWED', 
                 'log_AVG_MEAN_RATING', 'log_CONTACTS_W_CUSTOMER_SERVICE', 
                 'UNIQUE_MEALS_PURCH','log_AVG_PREP_VID_TIME', 'log_AVG_TIME_PER_SITE_VISIT', 
                 'Largest order size to Total Orders','High_Customer_Statisfaction'
                  'Largest order size to Total Pohots']


                                                                   

# preparing for scikit-learn

# Preparing a DataFrame based the the analysis above
x_data = apprentice_chef.loc[ : , x_var]


# preparing response variable
y_data      = apprentice_chef.loc[ : , 'REVENUE']
log_y_data  = apprentice_chef.loc[ : , 'log_REVENUE']


#################################
## setting up train-test split ##
#################################
x_train, x_test, y_train, y_test = train_test_split(
            x_data, # x-variables (can change this)
            log_y_data, # y-variable  (can change this)
            test_size    = 0.25,
            random_state = 219)



# In[166]:


# Setting a model name
model_name = "Linear Regression"


# INSTANTIATING a model object - CHANGE THIS AS NEEDED
model = sklearn.linear_model.LinearRegression()


# FITTING to the training data
model_fit = model.fit(x_train, y_train)


# PREDICTING on new data
model_pred = model.predict(x_test)


# SCORING the results
model_train_score = model.score(x_train, y_train).round(4) # using R-square
model_test_score  = model.score(x_test, y_test).round(4)   # using R-square
model_gap         = abs(model_train_score - model_test_score).round(4)


# displaying results
print('Training Score :', model_train_score)
print('Testing Score  :', model_test_score)
print('Train-Test Gap :', model_gap)


# ## Checking Model P-Value and Statistics 

# Checking the restults of P-Values below we find some interesting findingings. The P-values are used to determine the significance of each feature in predicting the dependent variable. A low P-value indicates that the feature is significant in predicting the dependent variable. In this case, it appears that mobile logins have a large P-value, indicating that it is not a significant predictor of revenue and thus will not be used in the model. Features with high P-values do not explain the variance and tend to cause noise in the model.
# Interpreting the coefficients, it is found that total meals ordered and average time per site have high coefficients, indicating a positive relationship with revenue. This means that increasing the number of meals ordered and the time spent on the site could potentially increase revenue.
# Additionally, it is important to investigate the user's interaction with the website This suggests that there may be some interesting findings related to the user experience on the website that could be explored to further increase revenue.
# The AIC and BIC are both low, indicating that the selected features are well justified and the model is a good fit. The model also has a high R-squared value of 67.3%, indicating that 67.3% of the variance in revenue can be explained by the predictors in the model. This suggests that the selected features are highly relevant in predicting revenue and that the model is reliable in making predictions.
# Further, the analysis suggests that total meals ordered and average time per site are significant predictors of revenue, and that the user experience on the website is an important factor to investigate further.
# 

# In[215]:


lm_fit = smf.ols(formula ="""log_REVENUE ~  log_TOTAL_MEALS_ORDERED +
                                            log_TOTAL_PHOTOS_VIEWED + 
                                            log_AVG_MEAN_RATING  + 
                                            log_CONTACTS_W_CUSTOMER_SERVICE + 
                                            UNIQUE_MEALS_PURCH + 
                                            log_AVG_PREP_VID_TIME + 
                                            log_AVG_TIME_PER_SITE_VISIT +  
                                            MOBILE_LOGINS +
                                            UNIQUE_MEALS_PURCH+
                                            log_CONTACTS_W_CUSTOMER_SERVICE""", 
                                                             data = apprentice_chef)

                        

# telling Python to run the data through the blueprint
results_fit = lm_fit.fit()


# printing the results
print(results_fit.summary())


# In[ ]:


# creating a (Pearson) correlation matrix
#High level view for which features are 
#Strong Correlated with Revenue
df_corr = apprentice_chef.corr().round(2)


# printing (Pearson) correlations with SalePrice
df_corr.loc[ : , ['REVENUE', 'log_REVENUE'] ].sort_values(by = 'REVENUE',
                                                                ascending = False)


# In[170]:


from sklearn.tree     import DecisionTreeRegressor     # regression trees
from sklearn.ensemble import RandomForestRegressor     # random forest
from sklearn.ensemble import GradientBoostingRegressor # gbm


# importing machine learning tools
from sklearn.model_selection import train_test_split # train-test split
from sklearn.tree import plot_tree                   # tree plots


# setting pandas print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 100)


# ## Ideal Model Used 

# In[201]:


#specifying a model name
model_name = 'Pruned Random Forest'


# INSTANTIATING a random forest model with default values
model = RandomForestRegressor(n_estimators     = 500,
                             criterion        = 'squared_error',
                             max_depth        = 12,
                             min_samples_leaf = 13,
                             bootstrap        = True,
                             warm_start       = True,
                             random_state     = 219)


# FITTING the training data
model_fit = model.fit(x_train, y_train)


# PREDICTING based on the testing set
model_pred = model.predict(x_test)


# SCORING the results
model_train_score = model.score(x_train, y_train).round(4) # using R-square
model_test_score  = model.score(x_test, y_test).round(4)   # using R-square
model_gap         = abs(model_train_score - model_test_score).round(4)


# displaying results
print('Training Score :', model_train_score)
print('Testing Score  :', model_test_score)
print('Train-Test Gap :', model_gap)

