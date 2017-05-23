

from yahoo_finance import Share
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
home = expanduser("~")

import os

import os
os.system('cls' if os.name == 'nt' else 'clear')
clf()
endDay =365*8
startDay = 1
startDate = date.today()-timedelta(days=endDay)
endDate = date.today()-timedelta(days=startDay)
#print (startDate, endDate)
#dates = pd.date_range(startDate, periods=60)
#print(dates)

def cdf(dev_divided_by_std):
    return (1.0 + math.erf(dev_divided_by_std / sqrt(2.0))) / 2.0


wti = Share('WTI')
uup = Share('UUP')
wtiHistory_raw = wti.get_historical(str(startDate), str(endDate))
uupHistory_raw = uup.get_historical(str(startDate), str(endDate))

datestrs = [startDate]*(len(wtiHistory_raw))
pricestrs_wti = [0.0]*(len(wtiHistory_raw))

for i in range(len(wtiHistory_raw)):
    datestrs[i] = wtiHistory_raw[i]['Date']
    pricestrs_wti[i] = float(wtiHistory_raw[i]['Close'])

pricestrs_uup = [0.0]*(len(wtiHistory_raw))
for i in range(len(uupHistory_raw)):
    pricestrs_uup[i] = float(uupHistory_raw[i]['Close'])

wtiHistory = pd.Series(pricestrs_wti, index = datestrs)
uupHistory = pd.Series(pricestrs_uup, index = datestrs)

df = pd.DataFrame({'wti':pricestrs_wti, 'uup':pricestrs_uup}, index = datestrs)
#mult = df.wti * df.uup
#df['multiplication'] = mult



model = linear_model.LinearRegression()

X= np.array(df.uup)
X_= np.array([np.ones(len(X)), X, X**2]).T
model.fit(X_, 1/np.array(df.wti))
model.coef_
X_=np.matrix(X_)
#plt.plot(X_*np.matrix(model.coef_).T, 1/np.array(df.uup))
#plt.show()
wti_fit = np.array(1/model.predict(X_))

df['regression'] = pd.Series(wti_fit, index = datestrs)

dev = df.wti-df['regression']
stdev = sqrt(mean(dev**2))
alpha = dev/stdev

df['alpha'] = pd.Series(alpha, index = datestrs)

print df

df.sort_index(inplace=True)
figure(num=1, figsize=(6,6))
#df[['wti', 'fit']].plot()
df[['wti', 'regression']][0:2000].plot()
plt.ylabel("WTI")
plt.xlabel("Date")
#plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
savefig(home+'/html/python/img/WTI.png', bbox_inches='tight')
#show()
show()
clf()







