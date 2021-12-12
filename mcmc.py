from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
import pymc3 as pm
import patsy as pt
import seaborn as sns
import os
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
import gc
import numpy as np
from mongo_test import get_data

import sys
import re

plt.style.use('seaborn-darkgrid')

plt.rcParams['figure.figsize'] = 14, 6
np.random.seed(0)
print('Running on PyMC3 v{}'.format(pm.__version__))

sale, calendar = get_data()
sales = pd.Series(sale.iloc[:, 2].values, index=sale['id'])
sales = pd.to_numeric(sales)

calendar['event_true_1'] = calendar.event_name_1.notna()
calendar['event_true_2'] = calendar.event_name_2.notna()

calendar['event_true_all'] = calendar.event_true_1 | calendar.event_true_2
calendar['event_true_all'] = calendar.event_true_all.apply(lambda x: x > 0)
calendar['event_true_all'] = calendar.event_true_all.astype('int')
calendar['date'] = pd.to_datetime(calendar.date)
calendar['d_parse'] = calendar.d.apply(lambda x: int(x.split('_')[1]))

calendar_feature = calendar[['wm_yr_wk', 'wday', 'month', 'year',
                             'snap_CA', 'snap_TX', 'snap_WI',
                             'event_true_all', 'd_parse']]

fml = 'total ~ wday + month + year + snap_CA + snap_TX + snap_WI + event_true_all + d_parse'

scaler = StandardScaler()

calendar_feature = calendar[['wm_yr_wk', 'wday', 'month', 'year',
                             'snap_CA', 'snap_TX', 'snap_WI',
                             'event_true_all', 'd_parse']]

scaled_feature = pd.DataFrame(scaler.fit_transform(calendar_feature))
scaled_feature.columns = calendar_feature.columns
scaled_feature.min()


np.where(sales < 10000)[0]

sales.iloc[[330,  696, 1061, 1426, 1791]
           ] = np.quantile(sales, 0.025)

df = scaled_feature.iloc[:1913, :]
df.loc[:, 'total'] = sales.values
df.loc[:, 'd_parse'] = calendar_feature.iloc[:1913, 8] - \
    np.min(calendar_feature.d_parse) + 1

(mx_en, mx_ex) = pt.dmatrices(fml, df, return_type='dataframe', NA_action='raise')
pd.concat((mx_ex.head(3), mx_ex.tail(3)))

print("CAL MODEL")

with pm.Model() as mdl_first:
    b0 = pm.Normal('b0_intercept', mu=0, sigma=1)
    b2 = pm.Normal('b2_wday', mu=0, sigma=1)
    b3 = pm.Normal('b3_month', mu=0, sigma=1)
    b4 = pm.Normal('b4_year', mu=0, sigma=1)
    b5 = pm.Normal('b5_snapCA', mu=0, sigma=1)
    b6 = pm.Normal('b6_snapTX', mu=0, sigma=1)
    b7 = pm.Normal('b7_snapWI', mu=0, sigma=1)
    b8 = pm.Normal('b8_event_true_all', mu=-0.01, sigma=1)

    theta = (b0 +
             b2 * mx_ex['wday'] +
             b3 * mx_ex['month'] +
             b4 * mx_ex['year'] +
             b5 * mx_ex['snap_CA'] +
             b6 * mx_ex['snap_TX'] +
             b7 * mx_ex['snap_WI'] +
             b8 * mx_ex['event_true_all'] +
             np.log(mx_ex['d_parse']))

    y = pm.Poisson('y', mu=np.exp(theta), observed=mx_en['total'].values)


print("DONE MODEL")

with mdl_first:
    trace = pm.sample(1000, tune=2000, init='adapt_diag',
                      target_accept=.8, cores=1)

mdl_first.check_test_point()
print("CHECKED")


def strip_derived_rvs(rvs):
    ret_rvs = []
    for rv in rvs:
        if not (re.search('_log', rv.name) or re.search('_interval', rv.name)):
            ret_rvs.append(rv)
    return ret_rvs


def plot_traces_pymc(trcs, varnames=None):
    nrows = len(trcs.varnames)
    if varnames is not None:
        nrows = len(varnames)

    ax = pm.traceplot(trcs, var_names=varnames, figsize=(12, nrows*1.4),
                      lines=tuple([(k, {}, v['mean'])
                                   for k, v in pm.summary(trcs, varnames=varnames).iterrows()]))

    for i, mn in enumerate(pm.summary(trcs, varnames=varnames)['mean']):
        ax[i, 0].annotate('{:.2f}'.format(mn), xy=(mn, 0), xycoords='data',
                          xytext=(5, 10), textcoords='offset points', rotation=90,
                          va='bottom', fontsize='large', color='#AA0022')


print("SUMMARY")
rvs_fish = [rv.name for rv in strip_derived_rvs(mdl_first.unobserved_RVs)]
pm.summary(trace, rvs_fish)

print("PLOTS")


pm.plot_trace(trace)
plt.show()
with mdl_first:
    pp_trace = pm.sample_posterior_predictive(
        trace, var_names=rvs_fish, samples=4000)
