# Вариант № 1-1-1

from statsmodels.stats.weightstats import ztest
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.ma.core import sqrt
from tqdm import tqdm

mu = 10
sigma = 5
alpha_value = 0.3
rv_first = stats.norm(loc=mu, scale=sigma)
rv_second = stats.laplace(1, 2)
number_of_samples = 100
margin = 0.0001
np.random.seed(42)
size = number_of_samples
sample_first = rv_first.rvs(size=size)
sample_second = rv_second.rvs(size=size)
plt.figure(figsize=(10, 5))
plt.hist(sample_first, bins=10)
plt.show()
plt.figure(figsize=(10, 5))
plt.hist(sample_second, bins=10)
plt.show()


def z_value_dm(a, b, n):
    z_val = (np.mean(a) - np.mean(b)) / (sqrt((np.var(a) + np.var(b)) / 2) * sqrt(2 / n))
    return z_val


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
plt.ylim((0, 1))
n = 25
m = 50

z_stat = 0
data = []
for i in range(m):
    data.append(rv_first.rvs(size=n))
calculated_stats = []
for i, a in tqdm(enumerate(data), total=m):
    for j, b in enumerate(data):
        if i != j:
            z_stat = z_value_dm(a, b, n)
            calculated_stats.append(z_stat)
sns.histplot(calculated_stats, ax=ax, bins=100, stat='density')
sns.kdeplot(calculated_stats, ax=ax, color='r', label='Приближение')

rv_theoretical = stats.t(df=(len(sample_first) + len(sample_second)
- 2))
line_x = np.linspace(rv_theoretical.ppf(margin), rv_theoretical.ppf
(1 - margin), number_of_samples)
sns.lineplot(x=line_x, y=rv_theoretical.pdf(line_x), color='b', lw=1, ax=ax, label='Распределение')

plt.show()


fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (15, 8))
plt.ylim((0,1))
n = number_of_samples
rv_theoretical = stats.t(df=(len(sample_first) + len(sample_second)
- 2))
local_alpha_value = (alpha_value / 2)
left_vline_position = rv_theoretical.ppf(local_alpha_value)
right_vline_position = rv_theoretical.ppf(1 - local_alpha_value)
line_x = np.linspace(rv_theoretical.ppf(margin), rv_theoretical.ppf
(1 - margin), number_of_samples)
sns.lineplot(x=line_x, y=rv_theoretical.pdf(line_x), color='b', lw=
1, ax=ax)
x = np.linspace(rv_theoretical.ppf(margin), rv_theoretical.ppf(1 -
margin), number_of_samples*10)
x_range = x[x<=left_vline_position]
ax.fill_between(x_range, rv_theoretical.pdf(x_range), np.zeros(len(
x_range)), color='b', alpha=0.4)
x = np.linspace(rv_theoretical.ppf(margin), rv_theoretical.ppf(1 -
margin), number_of_samples*10)
x_range = x[x>=right_vline_position]
ax.fill_between(x_range, rv_theoretical.pdf(x_range), np.zeros(len(
x_range)), color='b', alpha=0.4)

plt.show()

def z_value_dm(a, b, n):
    z_val = (np.mean(a) -
    np.mean(b)) / (sqrt((np.var(a) + np.var(b)) / 2) * sqrt(2/n))
    return z_val

rv_theoretical = stats.t(df=(len(sample_first) + len(sample_second)
- 2))
z_stat, p = stats.ttest_ind(sample_first, sample_second)
z_value_d = z_value_dm(sample_first, sample_second, 100)

print("Z =", z_stat)
print("Z =", z_value_d)
print("P_лев_критическое =", left_vline_position)
print("P_прав_критическое= ", right_vline_position)