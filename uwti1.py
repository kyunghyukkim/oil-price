from yahoo_finance import Share
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import datetime
from datetime import date, timedelta
from sklearn import linear_model
from pylab import *
from optparse import OptionParser
from pylab import *
from os.path import expanduser
from pandas.tseries.offsets import BDay
import os
import datetime, threading, time
from pandas.tseries.offsets import BDay
import json

home = expanduser("~")
os.system('cls' if os.name == 'nt' else 'clear')


########################################################################################
# Trading strategy
# 1. Momentum
# 2. uwti price vs. dollar value (training for 3 days) --> linear regression.
# 3. uwti price vs. dollar value (cross validation for 2 days) by using the linear model.
########################################################################################

# Slope from 3 day training data.
def compute_slope(df):
    y = np.array(df.uwti)
    x = np.array(range(0,len(df)))[:,np.newaxis]
    model.fit(x, y)
    return [model.coef_[0], model.intercept_]

########################################################################################
# 1. Momentum
########################################################################################

# If the slope is positive and large enough, then "consider" trading for the next two days.
def trading_condition(df):
    slope=compute_slope(df)[0]
    intercept = compute_slope(df)[1]
    #print slope

    sigma = np.std(df.uwti - 1/(intercept + slope * df.uup))
    #print sigma
    #if slope > 0:
    if slope > sigma/len(df):
        return True
    else:
        return False




# cumulative distribution function -- probability that the uwti stock price can increase.
# probability = alpha < 0.5 -- uwti is overestimated and likely goes down for the next few days.
# probability = alpha > 0.5 -- uwti is underestimated and likely increases for the next few days.
def cdf(dev_divided_by_std):
    return (1.0 + math.erf(dev_divided_by_std / sqrt(2.0))) / 2.0



def execute_trading(df, dfcv):

    ########################################################################################
    # 2. uwti price vs. dollar value (training for 3 days) --> linear regression.
    ########################################################################################

    #training set = df
    X = np.array(df.uup)
    X_= np.array([np.ones(len(X)), X]).T

    #cross validation set = dfcv
    Xcv = np.array(dfcv.uup)
    Xcv_ = np.array([np.ones(len(Xcv)),Xcv]).T

    #linear regression
    model.fit(X_, 1/np.array(df.uwti))
    model.coef_
    if model.intercept_ > 0:

        #standard deviation is computed from the training set and will be used to compute alpha.
        X_ = np.matrix(X_)
        uwti_fit = np.array(1/model.predict(X_))
        df['regression'] = pd.Series(uwti_fit, index = df.index)
        dev = df.uwti-df.regression
        stdev = sqrt(mean(dev**2))

        ########################################################################################
        # 3. uwti price vs. dollar value (cross validation for 2 days) by using the linear model.
        ########################################################################################

        #cross validation
        Xcv_ = np.matrix(Xcv_)
        uwti_fit = np.array(1/model.predict(Xcv_))
        dfcv['prediction'] = pd.Series(uwti_fit, index = dfcv.index)

        bought = False
        sold = False

        for i in range(0, len(dfcv)):
            dev = dfcv.uwti[i]-dfcv.prediction[i]
            alpha = dev/stdev
            short_pos = cdf(alpha)*100
            if bought != True and i != len(dfcv)-1:
                if short_pos < 10:
                    bought = True
                    date_bought = dfcv.index[i]
                    date_bought_index = i
                    value_bought = dfcv.uwti[i]
                    #print "Bought on %s at %f"%(str(date_bought), value_bought)
            elif bought == True:
                if i==len(dfcv)-1 or (dfcv.uwti[i]-value_bought)/value_bought<-0.02 or (dfcv.uwti[i]-value_bought)/value_bought > 0.05:
                    sold = True
                    date_sold = dfcv.index[i]
                    value_sold = dfcv.uwti[i]
                    #print "Sold on %s at %f"%(str(date_sold), value_sold)
                    return [True, 1+(value_sold-value_bought)/value_bought]
        if bought == True and sold == False:
            print "error!!!!"
            return [True, 1+(value_sold-value_bought)/value_bought]
        if bought == False:
            return [False, 1]
    else:
        return [False, 1]



# Cumulative distribution function
def cdf(dev_divided_by_std):
    return (1.0 + math.erf(dev_divided_by_std / sqrt(2.0))) / 2.0



# Retrieve real-time stock prices
# Data pre-processing
# Panda dataframe (df): main data file
startDate = date.today()-timedelta(days=10)
endDate = date.today()

uwti = Share('UWTI')
uup = Share('UUP')
uwtiHistory_raw = uwti.get_historical(str(startDate), str(endDate))
uupHistory_raw = uup.get_historical(str(startDate), str(endDate))

datestrs = [startDate]*(len(uwtiHistory_raw))
pricestrs_uwti = [0.0]*(len(uwtiHistory_raw))

for i in range(len(uwtiHistory_raw)):
    datestrs[i] = pd.to_datetime(uwtiHistory_raw[i]['Date'])
    pricestrs_uwti[i] = float(uwtiHistory_raw[i]['Close'])

pricestrs_uup = [0.0]*(len(uwtiHistory_raw))
for i in range(len(uupHistory_raw)):
    pricestrs_uup[i] = float(uupHistory_raw[i]['Close'])

uwtiHistory = pd.Series(pricestrs_uwti, index = datestrs)
uupHistory = pd.Series(pricestrs_uup, index = datestrs)

df = pd.DataFrame({'uwti':pricestrs_uwti, 'uup':pricestrs_uup}, index = datestrs)
df.sort_index(inplace=True)
df = df[-4:] # stock price data for the last 4 days were used as training data.


###############################################################################
# Linear regression: y = a0 + a1 x, where y = 1/uwti and x = uup (dollar value)
###############################################################################

executed = False
trainingPeriod = 4 # training period = last 4 days (closing price)

df.sort_index(inplace=True)

if trading_condition(df[['uup', 'uwti']][0:trainingPeriod]):
    [executed, gain] = execute_trading(df[['uup', 'uwti']][0:4], df[['uup', 'uwti']][trainingPeriod+i:trainingPeriod+cvPeriod+i])

model = linear_model.LinearRegression()
X= np.array(df.uup)
X_= np.array([np.ones(len(X)), X]).T
model.fit(X_, 1/np.array(df.uwti))
model.coef_
X_=np.matrix(X_)
uwti_fit = np.array(1/model.predict(X_))
df['regression'] = pd.Series(uwti_fit, index = df.index)


def foo():
    time_out = time.time() + 3600*20
    while True:
        # Trading suggestion based on the current uwti and uup and the previous 5 days data (regression).
        if time.time() > time_out:
            break

        dev = df.uwti-df['regression']
        stdev = sqrt(mean(dev**2))
        uwti_now = float(uwti.get_price())
        uup_now = float(uup.get_price())
        regression_now = 1/model.predict([1, uup_now])
        dev_now = uwti_now - regression_now
        alpha_now = dev_now/stdev
        df_all = df.append(pd.DataFrame(data = {'uwti':uwti_now, 'uup':uup_now, 'regression':regression_now}, index = [pd.to_datetime(uwti.get_trade_datetime())]))

        # Training set (regression) + prediction (regression_now)
        font = {'weight' : 'bold',
                'size'   : 14}
        matplotlib.rc('font', **font)
        ax = df_all[['uwti', 'regression']].plot(marker='o', linestyle='--')
        ax.set_ylabel("UWTI", fontsize=20)
        ax.set_xlabel("Date", fontsize=20)
        plt.fill_between(pd.to_datetime(df_all.index[0:5]),1.3, 2.0, facecolor='green', alpha=0.1)
        savefig('/var/www/html/python/img/UWTI.png', bbox_inches='tight')
        plt.show()

        # UWTI short and long pos probabilities (%)
        short_pos = cdf(alpha_now)*100
        long_pos = 100-short_pos

        prob = {"cols":[{"label":"Position","type":"string"}, {"label":"Probability","type":"number"}],
                "rows":[{"c":[{"v": "Short"}, {"v":short_pos}]},{"c":[{"v": "Long"}, {"v":long_pos}]}]
        }
        with open('/var/www/html/python/img/uwti.json', 'wb') as outfile:
            json.dump(prob, outfile)

        # DWTI short and long pos probabilities (%)
        prob = {
            "cols":[{"label":"Position","type":"string"}, {"label":"Probability","type":"number"}],
            "rows":[{"c":[{"v": "Short"}, {"v": long_pos}]}, {"c":[{"v": "Long"}, {"v": short_pos}]}]
        }
        with open('/var/www/html/python/img/dwti.json', 'wb') as outfile:
            json.dump(prob, outfile)
        #clf()

        #uwti_analysis_now = { str(datetime.datetime.now()):{"Price":uwti_now, "Regression":float(regression_now)}}
        uwti_analysis_now = {"Price":uwti_now}
        with open('/var/www/html/python/img/uwti_analysis_now.json', 'wb') as outfile:
            json.dump(uwti_analysis_now, outfile)

        time.sleep(180)



timerThread = threading.Thread(target=foo)
timerThread.start()


#########################
## google chart pie plots
#########################
# font = {'weight' : 'bold',
#         'size'   : 22}
# matplotlib.rc('font', **font)
# fig2 = figure(num=2, figsize=(6,6))
#
# stock = ('DWTI','UWTI')
# r = len(stock)
# ax = axes([0.1, 0.1, 0.8, 0.8])
#
# labels = 'Short', 'Long'
# fracs = [short_pos, long_pos]
# explode=(0, 0.05)
# pie(fracs, explode=explode, labels=labels, autopct='%1.0f%%', shadow=True, startangle=90,
#     colors = cm.Set1([0.2,0.5]))
# title('UWTI')
# savefig('/var/www/html/python/img/UWTI_prob.png', bbox_inches='tight')
# #clf()


# # make a square figure and axes
# plt.figure(num=3, figsize=(6,6))
# ax = axes([0.1, 0.1, 0.8, 0.8])
# font = {
#         'weight' : 'bold',
#         'size'   : 22}
# matplotlib.rc('font', **font)
# labels = 'Short', 'Long'
# fracs = [long_pos, short_pos]
# explode=(0, 0.05)
# pie(fracs, explode=explode, labels=labels, autopct='%1.0f%%', shadow=True, startangle=90,
#     colors = cm.Set1([0.2,0.5]))
# title('DWTI')
# savefig('/var/www/html/python/img/DWTI_prob.png', bbox_inches='tight')
#clf()
