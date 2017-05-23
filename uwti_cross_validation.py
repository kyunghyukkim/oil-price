from yahoo_finance import Share
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import date, timedelta
from sklearn import linear_model
from pylab import *
from optparse import OptionParser
from os.path import expanduser
import os

home = expanduser("~")
os.system('cls' if os.name == 'nt' else 'clear')

font = {'weight' : 'bold',
        'size'   : 14}
matplotlib.rc('font', **font)


# Cumulative distribution function
def cdf(dev_divided_by_std):
    return (1.0 + math.erf(dev_divided_by_std / sqrt(2.0))) / 2.0


######################################
# Retrieve real-time stock prices
# Data pre-processing
# Panda dataframe (df): main data file
######################################

###################################################
# Retrieve historical stock data for uwti, wti, uup
###################################################
def retrieve_stock_data(stock):
    # look for last 8 year data if available
    endDay =365*8
    startDay = 1

    startDate = date.today()-timedelta(days=endDay)
    endDate = date.today()-timedelta(days=startDay)

    #retrieve "stock" data.
    uwti = Share(stock)
    #uup: stock showing the relative dollar value
    uup = Share('UUP')
    uwtiHistory_raw = uwti.get_historical(str(startDate), str(endDate))
    uupHistory_raw = uup.get_historical(str(startDate), str(endDate))

    datestrs = [startDate]*(len(uwtiHistory_raw))
    pricestrs_uwti = [0.0]*(len(uwtiHistory_raw))

    if len(uwtiHistory_raw)>len(uupHistory_raw):
        max = len(uupHistory_raw)
    else:
        max = len(uwtiHistory_raw)

    for i in range(max):
        datestrs[i] = uwtiHistory_raw[i]['Date']
        pricestrs_uwti[i] = float(uwtiHistory_raw[i]['Close'])

    pricestrs_uup = [0.0]*max

    for i in range(max):
        pricestrs_uup[i] = float(uupHistory_raw[i]['Close'])

    datestrs = pd.to_datetime(datestrs)
    df = pd.DataFrame({stock.lower():pricestrs_uwti, 'uup':pricestrs_uup}, index = datestrs)
    df.sort_index(inplace=True)
    print "\n\n\n"
    print "#################################################"
    print "Historical data for %s is successfully retrieved." % stock
    print "#################################################"
    return df



######################################################################################
# Linear regression over segmented regions of time series for different segment sizes.
######################################################################################

def cc_seg(interval, df, RenderPlot):
    model = linear_model.LinearRegression()

    dfseg_all = pd.DataFrame()
    for i in range(interval, len(df)//interval*interval, interval):
        dfseg = df[['uwti','uup']][i:i+interval]

        Xseg = np.array(dfseg.uup)

        # Different linear polynomial models were tested.
        Xseg_= np.array([np.ones(len(Xseg)), Xseg]).T
        #Xseg_= np.array([np.ones(len(Xseg)), Xseg, Xseg**2]).T
        #Xseg_= np.array([np.ones(len(Xseg)), Xseg, Xseg**2, Xseg**3]).T
        model.fit(Xseg_, 1/np.array(dfseg.uwti))
        model.coef_
        Xseg_ = np.matrix(Xseg_)
        uwti_fit_seg = np.array(1/model.predict(Xseg_))

        dfseg['regression'] = pd.Series(uwti_fit_seg, index = df.index[i:i+interval])
        dfseg_all = dfseg_all.append(dfseg)
    if RenderPlot:
        ax = dfseg_all[['regression', 'uwti']].plot()
        ax.set_xlabel('Date', fontsize = 20)
        ax.set_ylabel('UWTI', fontsize = 20)
        ax.text(0.8,0.7,r'$\Delta$ Trading Day = ' + str(interval),
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform = ax.transAxes)
        dfseg_all['num']=pd.Series(range(1,len(dfseg_all)+1), index = dfseg_all.index)
        plt.fill_between(pd.to_datetime(dfseg_all.index),0.1, dfseg_all.uwti, where=dfseg_all.num%(interval*2)>interval, facecolor='green', alpha=0.1)
        plt.show()

    return np.corrcoef(dfseg_all['regression'],dfseg_all['uwti'])[0][1]



############################################
# main commands start from here.
############################################

# Retrieve stock data (WTI)
# The reason that we did not pull out uwti is that the historical data for wti is much larger than uwti.
# Prediction, cross-validation, training -- we will use uwti.

df = retrieve_stock_data('WTI')



# Linear regression: y = a0 + a1 x, where y = 1/wti and x = uup (dollar value)
model = linear_model.LinearRegression()

X= np.array(df.uup)
X_= np.array([np.ones(len(X)), X]).T
model.fit(X_, 1/np.array(df.wti))
model.coef_
X_=np.matrix(X_)
wti_fit = np.array(1/model.predict(X_))
df['regression'] = pd.Series(wti_fit, index = df.index)



# Inverse relationship between wti and uup.
# Potential time scale issue.
df[['wti', 'regression']].plot()
plt.ylabel(r'WTI', fontsize = 20)
plt.xlabel(r'Year', fontsize = 20)
plt.show()
plt.savefig('/var/www/html/python/img/WTI.png')


# Now, we retrieve uwti data instead of wti.
df = retrieve_stock_data('UWTI')

# Linear regression is performed over segmented time series data (uwti vs. date).
# The segment size is "interval".
# For different sizes of "interval", we will investigate how well the linear regression (piece-wise) is performed.

# First, let's look at the cases of three different time intervals, [10, 30, 120]

interval = 120
print 'correlation coefficient = %f' % cc_seg(interval, df, True)

interval = 30
print 'correlation coefficient = %f' % cc_seg(interval, df, True)

interval = 10
print 'correlation coefficient = %f' % cc_seg(interval, df, True)


# Second, let's compute correlation coefficients between uwti and its regression curve for different interval values.
# cc_trend = correlation coefficients.
intervals = range(5,100,4)

cc_trend1 = [cc_seg(interval, df, False) for interval in intervals]
cc_trend2 = [cc_seg(interval, df[interval//3*1:], False) for interval in intervals]
cc_trend3 = [cc_seg(interval, df[interval//3*2:], False) for interval in intervals]

cc = pd.DataFrame()
cc['sample1'] = pd.Series(cc_trend1, index = intervals)
cc['sample2'] = pd.Series(cc_trend2, index = intervals)
cc['sample3'] = pd.Series(cc_trend3, index = intervals)

cc['average'] = cc.mean(axis=1)

# correlation coefficient vs. time interval.
plt.figure(num=10, figsize=(6,6))
cc['average'].plot(marker='o', linestyle='--')
plt.ylabel("Correlation coefficient", fontsize = 20)
plt.xlabel(r"$\Delta$ Trading Day", fontsize = 20)





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
    w1=compute_slope(df)[0]
    intercept = compute_slope(df)[1]
    #print slope

    sigma = np.std(df.uwti - 1/(intercept + w1 * df.uup))
    slope = -1/(intercept+w1*np.mean(df.uup))**2 * w1 * np.mean(df.uup)

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
    # 2. uwti price vs. dollar value (training for 4 days) --> linear regression.
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



#####################################################################
# Plot the net gain over time.
#####################################################################

gain=0
gain_time = 1
gain_time_list = []
executed = False
cvPeriod = 2
trainingPeriod = 4

df.sort_index(inplace=True)

for i in range(0,len(df)-trainingPeriod-cvPeriod, cvPeriod):
    if trading_condition(df[['uup', 'uwti']][i:trainingPeriod+i]):
        [executed, gain] = execute_trading(df[['uup', 'uwti']][i:trainingPeriod+i], df[['uup', 'uwti']][trainingPeriod+i:trainingPeriod+cvPeriod+i])
        #print executed, gain
        gain_time *= gain
    gain_time_list.append(gain_time)

plt.figure(num=20)
plt.plot(np.array(gain_time_list))
plt.xlabel("Days", fontsize = 20)
plt.ylabel("Net Gain", fontsize = 20)
plt.savefig('/var/www/html/python/img/gain.png')






