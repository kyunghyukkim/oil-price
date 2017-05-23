

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
from pandas.tseries.offsets import BDay
import json

matplotlib.use('Agg')

import datetime, threading, time

import os
os.system('cls' if os.name == 'nt' else 'clear')

startDate = date.today()-timedelta(days=30)
#endDate = date.today()-timedelta(days=1)
endDate = date.today()
#print (startDate, endDate)
#dates = pd.date_range(startDate, periods=60)
#print(dates)

def cdf(dev_divided_by_std):
    return (1.0 + math.erf(dev_divided_by_std / sqrt(2.0))) / 2.0


uwti = Share('UWTI')
uup = Share('UUP')
uwtiHistory_raw = uwti.get_historical(str(startDate), str(endDate))
uupHistory_raw = uup.get_historical(str(startDate), str(endDate))

datestrs = [startDate]*(len(uwtiHistory_raw))
pricestrs_uwti = [0.0]*(len(uwtiHistory_raw))

for i in range(len(uwtiHistory_raw)):
    datestrs[i] = uwtiHistory_raw[i]['Date']
    pricestrs_uwti[i] = float(uwtiHistory_raw[i]['Close'])

pricestrs_uup = [0.0]*(len(uwtiHistory_raw))
for i in range(len(uupHistory_raw)):
    pricestrs_uup[i] = float(uupHistory_raw[i]['Close'])

uwtiHistory = pd.Series(pricestrs_uwti, index = datestrs)
uupHistory = pd.Series(pricestrs_uup, index = datestrs)

df = pd.DataFrame({'uwti':pricestrs_uwti, 'uup':pricestrs_uup}, index = datestrs)
#mult = df.uwti * df.uup
#df['multiplication'] = mult



model = linear_model.LinearRegression()

X= np.array(df.uup)
X_= np.array([np.ones(len(X)), X, X**2]).T
model.fit(X_, 1/np.array(df.uwti))
model.coef_
X_=np.matrix(X_)
#plt.plot(X_*np.matrix(model.coef_).T, 1/np.array(df.uup))
#plt.show()
uwti_fit = np.array(1/model.predict(X_))

df['regression'] = pd.Series(uwti_fit, index = datestrs)

dev = df.uwti-df['regression']
stdev = sqrt(mean(dev**2))
alpha = dev/stdev

df['alpha'] = pd.Series(alpha, index = datestrs)

print df

df.sort_index(inplace=True)

#show()
#show()
#clf()




print df


def foo():
    time_out = time.time() + 3600*14
    while True:
        if time.time() > time_out:
            break

        uwti_price_now = float(uwti.get_price())
        uup_price_now = float(uup.get_price())
        uwti_price_prediction = 1/model.predict( [1, uup_price_now, uup_price_now**2])
        dev_now = uwti_price_now - uwti_price_prediction
        alpha_now = dev_now/stdev

        #UWTI short or long pos probability (%)
        #short_pos = cdf(df.alpha[str(df.index.values[len(df)-1])])*100
        short_pos = cdf(alpha_now)*100
        long_pos = 100-short_pos

        prob = {"cols":[{"label":"Position","type":"string"}, {"label":"Probability","type":"number"}],
                "rows":[{"c":[{"v": "Short"}, {"v":short_pos}]},{"c":[{"v": "Long"}, {"v":long_pos}]}]
                }

        with open('uwti.json', 'wb') as outfile:
            json.dump(prob, outfile)

        # make a square figure and axes
        prob = {
            "cols":[
            {"label":"Position","type":"string"}, {"label":"Probability","type":"number"}
            ],
            "rows":[
                {"c":[{"v": "Short"}, {"v":long_pos}]},
                {"c":[{"v": "Long"}, {"v":short_pos}]}
            ]
        }
        with open('dwti.json', 'wb') as outfile:
            json.dump(prob, outfile)
        #clf()

        uwti_analysis_now = { str(datetime.datetime.now()):{"Price":uwti_price_now, "Regression":uwti_price_prediction}}

        with open('uwti_analysis_now.json', 'wb') as outfile:
            json.dump(uwti_analysis_now, outfile)

        time.sleep(60)


timerThread = threading.Thread(target=foo)
timerThread.start()
