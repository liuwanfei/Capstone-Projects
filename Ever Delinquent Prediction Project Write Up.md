
<h1><center>Mortgage Loan's Ever Delinquent Probability Prediction</center></h1>

<center>Sunny Liu</center>
<center>Dec 23, 2017</center>

## 1. Introduction
### 1.1 Problem Statement
Private mortgage insurance, or PMI, is typically required with most conventional (non-government backed) mortgage programs when the down payment is less than 20% of the property value. In other words, when purchasing or refinancing a home with a conventional mortgage, if the loan-to-value (LTV) is greater than 80% (or equivalently, the equity position is less than 20%), the borrower will likely be required to carry private mortgage insurance.

Company A is one of the top 5 players in PMI industry and is also the business target this project is focusing on. Similar to other insurance companies, the way company A makes money is by collecting PMI premium from the lenders (ultimately from home buyers). However, if a home buyer misses multiple mortgage payments, the loan will fall under "default/delinquent" status. If it is never cured within required time frame, the lender will file a claim to company A which pays agreed-upon coverage amount if the claim gets ultimately approved. 

The goal of this project is to predict mortgage loan's ever-delinquent probability by using machine learning approach, and assist company A make decisions on whether to insure a given mortgage loan sent from the lender.


### 1.2 Metrics
The key metrics used in this project to determine whether company A insures a loan are expected premium received amount and expected claim paid amount. See the calculation of each below:

- **Expected Premium Received** = Premium Amount **X**  Payment Frequency **X**  Expected Life of PMI <br>
- **Expected Claim Paid** = Loan Amount **X** Coverage % **X** **_Ever Delinquent Probability_** **X** Claim Rate

- If Expected Premium Received >= Expected Claim Paid, then insure the loan <br>
- If Expected Premium Received < Expected Claim Paid, then DO NOT insure the loan

### 1.3 Key Industry Terminologies
The followings are industry jargons mentioned in the paper. 
- **LTV**: Loan to Value. It represents the mortgage portion of the property. 
- **DTI**: Debt to Income ratio. The number is one way lenders measure your ability to manage the payments you make every month to repay the money you have borrowed. <br>
- **FICO**: Credit score used in US. The number represents the creditworthiness of a person, the likelihood that person will pay his or her debts. <br>
- **Loan Purpose**: It is a term in United States mortgage industry to show the underlying reason an applicant is seeking a loan. The purpose of the loan is used by the lender to make decisions on the risk and may even impact the interest rate that is offered. Possible loan purposes are "Purchase", "Refinance with Cash-Out" (higher risk) and "Refinance Pay-off Existing Lien".
- **Property Type**: Types of property is also one of the factors when considering default risks. For instance, company A does not insure investment property housing since 2008. Property types that company A insures are "Single-Family Home", "Condo" and "Manufactured Housing".
- **Origination Channel**: The channel which a mortgage loan is originated from are "Retail", "Broker" or "Correspondent".
 - *Retail:* A mortgage loan for which the mortgage loan seller takes the mortgage loan application and then processes, underwrites, funds and delivers the mortgage loan to Fannie Mae.
 - *Correspondent:* A mortgage loan that is originated by a party other than a mortgage loan seller and is then sold to a mortgage loan seller.
 - *Broker:* A mortgage loan that is originated under circumstances where a person or firm other than a mortgage loan seller or lender correspondent is acting as a “broker” and receives a commission for bringing together a borrower and a lender.
- **Occupancy Status**: The status includes "Principal Residence", "Second Home" and "Investment".
 - *Principal Residence:* A principal residence is a property that the borrower occupies as his or her primary residence.
 - *Second Home:* Second home is a property that the borrower occupies as his or her secondary residence.
 - *Investment:* An investment property is owned but not occupied by the borrower. 

## 2. Data
### 2.1 Data Description and Wrangling
**1) Data Population:** The data used in this project are mortgage loans insured by company A and were originated during 2010-2013. I purposely avoided any pre-2009 loans in order to minimize the impact from 2008 financial crisis. Using 2013 as the upper limit cut-off is based on the consideration of giving loans enough time (more than 4 years) to stabilize the status. <br>
**2) Data Size:** The total number of records for the above-mentioned population is 478,262. <br>
**3) Data Elements:** 27 data elements are included in the dataset. Some key loan characteristics are FICO, LTV, DTI, Occupancy Status, Loan Purpose, First-time Home Buyer Indicator, Number of Borrowers, Number of Units, Property Type and Origination Channel. I also included loan amount and premium related information for later dollar amount calculation purpose.<br>
**4) Missing Data:** If the missing data belongs to FICO, LTV or DTI, it gets filled with the most risky bucket to be conservative. For other data elements, the missing data gets replaced with the value based on "majority rule". <br> 
**5) Data Bucketing:** FICO, LTV and DTI values have been transformed into buckets in the dataset following the historically established bucketing rules.

### 2.2 Exploratory Data Analysis
Although there are 27 data elements originally included in the dataset, only **10** are selected as **features** mainly due to one of following reasons: <br>
1) The data element does not represent unique record ID in the dataset. <br>
2) Number of distinct values under a data element should be more than 1. In addition, the distribution of distinct values should not be extremely unbalanced (eg. if more than 99.5% of records share the same value, then the corresponding data element should be excluded). <br>
3) Should make business sense when considering the possible impact/cause to loan's ever delinquent status. <br>

*See below regarding selected feature's distribution as well as the target (ever delinquent flag) distribution*


| Orig Yr| % to All   |Claim %|
|--------------|-----------|---------|
|2010        |   10.6%   |0.6%
|2011       |   15.0%   |0.3%
|2012       |   34.5%   |0.1%
|2013       |   40.0%   |0.1%

| Loan Purp     | % to All  
|--------------|-----------
| Refi Cash Out       |   2.6%   
| Refi Payoff Lien      |  32.3%   
| Purchase    |  65.2%   

| Property Type     | % to All  
|--------------|-----------
| Co-op or Condo       |   9.8%   
| Manufacutre Housing    |  0.3%   
| Single Fam    |  89.9%   

| Occupancy Status     | % to All  
|--------------|-----------
| Primary Resident     |   96.5%   
| Secondary Resident    |  3.5% 

| First Time Home Buyer     | % to All  
|--------------|-----------
| Y     |   31.7%   
| N    |  68.3% 

| Multi Borrower    | % to All  
|--------------|-----------
| Y     |   49.2%   
| N    |  50.8% 

| MI Channel    | % to All  
|--------------|-----------
| Delegated     |   66.7%   
| Non-Delegated    |  33.3% 


| Ever Delinquent    | % to All  
|--------------|-----------
| Y     |   2.7%   
| N    |  97.3%
<br>
Features such as FICO, LTV and DTI have been transformed into corresponding buckets. The distribution of each is as below: <br>
![my_image](files/FICO Distr.jpg)
![my_image](files/LTV Distr.jpg)
![my_image](files/DTI Distr.jpg)

In addition, due to unbalanced class distribution of the target (2.7% delinquent loans vs. 97.3% never delinquent loans), I downsampled the "never delinquent" population. At end of the data exploration step, 10,000 loans were sampled for modeling: **5,000 never delinquent and 5,000 ever delinquent**.

## 3. Training and Modeling
### 3.1 Objective
**1) Target:** The target of this project is a given mortgage loan's ever delinquent probability. <br>
**2) Features:** 10 chosen features are: FICO, DTI, LTV, Origination Year, Loan Purpose, Property Type, Number of Borrower, Origination Channel, First Time Homebuyer Indicator, Occupancy Status. <br>
**3) Loss Functions:** <br>
- In machine learning, loss function is used to measure the degree of fit. It represents the price paid for inaccurately predicting a class(s).
- Below are some commonly used loss functions for classification problems: <br>
 
   [**_Logistic Loss:_**](http://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html) Log Loss measures the performance of a classification model whose output is a probability value between 0 and 1. The loss increases as the predicted probability diverges from the actual label. For binary classification problem, the log loss function is $−(y\cdot log(p)+(1 − y)\cdot log(1 − p))$. The example of algorithm which is based on log loss function is Logistic Regression. <br>
   **_Hinge Loss:_** The hinge loss is used for "maximum-margin" classification, most notably for support vector machines (SVMs). For an intended output t = ±1 and a classifier score y, the hinge loss of the prediction y is defined as $ℓ ( y ) = max ( 0 , 1 − t ⋅ y ) $. <br>
   [**_0/1 Loss:_**](https://stats.stackexchange.com/questions/284028/0-1-loss-function-explanation) This is one of the loss function which does not have convex shape $\min_\theta\sum_i L_{0/1}(\theta^Tx)$. We define $L_{0/1}(\theta^Tx) =1$ if $y = \theta^Tx$, and $L_{0/1}(\theta^Tx) =0$ if $y \neq\theta^Tx$. <br>

Since the target of this project is to predict probability, I tried different classification algorithms based on loss functions such as log loss or hinge loss.

### 3.2 Model Selection
**1) Classification Models:** Since this project is a typical classification problem, I tried 8 different classification models on the training set during "Model Selection" step. The models are Logistic Regression(LR), Linear Discriminant Analysis(LDA), K-NN, Decision Tree(CART), Naive Bays(NB), SVM, Kernel SVM and Random Forest(RF). <br>
![my_image](files/8 Models Selection.jpg) <br>
**2) ROC Curve and ROC AUC Score:** **_ROC Curve_** is the plot of TPR (True Positive Rate) and FPR (False Positive Rate) when comparing predicted target outcome vs. actual considering all possible 0 to 1 threshold. **_ROC AUC_** is the area under the ROC curve which represents the performance of the classifier. If **_ROC AUC score_** is 0.5 then the prediction is as good as random guessing; if ROC AUC score is 1 then prediction is perfect. The more ROC AUC score is closer to 1, the better the model prediction is. <br> <br>Below is the comparison among above-mentioned 8 classification models in terms of the mean of ROC AUC score as well as the standard deviation of ROC AUC score after cross validation step: <br>
![my_image](files/ROC AUC for All Models Code.JPG) <br>
![my_image](files/ROC AUC for All Models.JPG) <br>
<br>
As the boxplot shows, **_LR outperforms all other models_** - it has the highest mean of ROC AUC score as well as lowest standard deviation of ROC AUC score. LDA and SVM have the 2nd and 3rd best performance based on ROC AUC score compared to other non-linear models. <br> 

It is also noticeable that LR and LDA have very similar performance. This is because they have no difference in model function but assumptions on feature distribution and the estimation of the coefficients. In general, LR is the more flexible and more robust method in case of violations of these assumptions.
<br><br>
**3) Hyperparameter Optimization and Regularization:** <br>
 - **_Hyperparameter Optimization:_** This is an important step during model selection. The same kind of machine learning model can require different constraints, weights or learning rates to generalize different data patterns. These measures are called hyperparameters, and have to be tuned so that the model can optimally solve the machine learning problem. <br>
 - **_Regularization:_** Regularization refers to the method of preventing overfitting, by explicitly controlling the model complexity. It leads to smoothening of the regression line and thus prevents overfitting. It does so by penalizing the bent of the regression line that tries to closely match the noisy data points. There are couple of techniques to achieve regularization such as L1 and L2 based on different cost functions.<br><br> The step below shows how to get the best regularization technique as well as the best hyperparameter C selected for the LR model. <br>
![my_image](files/Parameter Optimization.JPG) <br>
**4) Model Calibration:** Since the goal of this project is to predict the probability of ever delinquency on a given loan, we want to evaluate how closely the model outcome (probability score) and the actually predicted probability. This is how calibration comes into play. It is used to improve probability estimation or error distribution of an existing model. Below is the calibration plot of Logistic Regression based on test dataset.  <br> 
![my_image](files/Calibration Plot Balanced.JPG) <br>
The plot tells us that the model is well-calibrated. In order words, we could use the probability score calculated from the model as the probability of target class - "ever delinquency" in this project.

### 3.3 Model Evaluation
**1) Model Coefficients:** The coefficients are helpful to provide high level direction on what features are more sensitive and significant in terms of impacting model predictions. As shown below, we see high DTI bucket, low FICO score and high LTV bucket are likely to drive a loan getting into delinquent status. This can be verified from the business perspectives as well.<br> 
![my_image](files/Feature Coefficience.JPG) <br>
**2) Training vs. Test ROC Curve Comparison:** This step is used to evaluate if the training set and test set have similar prediction performance using the trained model which is LR in this project. The ROC curve comparison graph below shows that the model has stable outcomes on both training and test set because the two ROC curves overlap each other. <br>
![my_image](files/Train vs Test ROC Curve.JPG) <br>
**3) PDF(Probability Density Function) vs. Prediction Score :** The graph below is a visual representation on how each class is separated by the probability score. For instance, based on the model outcome, if the probability score is 0.5, we have equally chance of predicting a loan going into delinquent and not going into delinquent status. As the score move towards 1, we predict a loan has higher chance of going into "delinquent" compared to "not delinquent". The overlap area of the plot explains where false categorizations fall into.  
![my_image](files/Class KDE Plot.JPG) <br>
**4) Business Evaluation: ** The key of building a model is to make the model useful and solve real world problems. In this project, the goal is to get a given loan's ever delinquent probability hence to calculate expected claim payment. Along with the calculated expected premium received, the business could evaluate if a loan is worth (making money) to insure. See calculated function below:<br>
- **Expected Premium Received** = Premium Amount **X**  Payment Frequency **X**  Expected Life of PMI <br>
- **Expected Claim Paid** = Loan Amount **X** Coverage % **X** **_Ever Delinquent Probability_** **X** Claim Rate

- If Expected Premium Received >= Expected Claim Paid, then insure the loan <br>
- If Expected Premium Received < Expected Claim Paid, then DO NOT insure the loan

Two assumptions here in the calculation: <br>
- Assume PMI lasts about 3 years for every MI eligible loan - from the time the PMI is activated till it is canceled. This is based on the historical data along with applying weighted average method on loan's PMI duration.
- Assume the claim rate on a delinquent loan is 8.5%. <br>

By applying the calculation method and assumptions above, we can conclude that 86.4% of test set population have expected premium payment exceeding expected claim payment and 13.6% of population have expected premium payment less than expected claim payment. In order words, although company A is insuring the 13.6% of the population at the moment, it will help company save more money if they choose not insure those loans. 

As for the equivalent dollar amount, company A is expectedly making 8.51 million insuring all the loans on its book. However, if excluding loans which have expected claim amount exceeding expected premium collected, company A could save 0.33 million, hence expectedly making 8.84 million in total as the revenue. The model actually results in 3.8% additional revenue.<br>
![my_image](files/expected premium and expected claim code.JPG)

## 4. Conclusions
- After all the steps mentioned above, from Data Exploratory Analysis, to Model Selection, and then to Model Evaluation, it appears that Logistic Regression is the best fitted algorithm for this machine learning problem.
- Selecting multiple features instead of using only FICO score improved the model prediction power based on the AUC score increasing from 0.65 to 0.74. This also makes business sense since there is more than 1 feature which impact the loan's ever delinquent probability prediction.
- The predictive model created in this project helps company A determine whether it costs money to insure a mortgage loan. In other words, if the expected claim payment is more than expected premium collected, company A is making a loss insuring the given loan. 
- As a result, if applying the predictive model created in this project, company A could potentially prevent 0.33 million dollars loss from the overall revenue 8.84 million dollars.   


## 5. Next Steps/Future Works
- **_Include More Features:_** Since mortgage loans' delinquency status can be caused by both loan characteristics and borrower behavior, there might be additional features which help model prediction power. So one of the future works is to collect data on new features such as borrower credit history, borrower occupancy, location of the house, self-employed Y/N etc.   
- **_Investigate RF and CART Performance:_** Investigate on why Random Forest and Decision Tree models underperform linear classification models in this project. 
- **_Redefine Target:_** In the real world, mortgage loans' have different level of delinquent status. For example, some loans might only miss one month payment or 30-day delinquent, some might miss four months payment or 120-day delinquent. Currently I am treating all level of delinquency the same, should we treat them differently? Will that improve the predicting power? 
- **_Improve Assumption under Metric Calculation:_** The current assumption in calculating expected claim payment is using 8.5% as the expected claim rate applying on all ever-delinquent loans. But in reality, claim rate defers based on level of delinquency. The more payments missed or more days of delinquency a loan has, the higher chance the loan will go from delinquent status to claim.   
