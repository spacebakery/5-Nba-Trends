import numpy as np
import pandas as pd
from scipy.stats import pearsonr, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

np.set_printoptions(suppress=True, precision = 2)

nba = pd.read_csv('./nba_games.csv')

# Subset Data to 2010 Season, 2014 Season
nba_2010 = nba[nba.year_id == 2010]
nba_2014 = nba[nba.year_id == 2014]

print(nba_2010.head())
print(nba_2014.head())

# task 1
knicks_pts_2010 = nba_2010[nba.fran_id == 'Knicks'].pts
nets_pts_2010 = nba_2010.pts[nba.fran_id == 'Nets']

# task 2
# diff of knicks mean scores and nets mean scores
diff_means_2010 = knicks_pts_2010.mean() - nets_pts_2010.mean()
print(diff_means_2010)

print('According to the mean value difference of less than 10 pts. We can state that the \
\nvariables (points and fran_id) do not have a significant association with one another.')

'''
- Answer: According to the mean value difference of less than 10 pts. We can state that the variables (points and fran_id) do not have a significant association with one another.
'''

# task 3
plt.hist(x=knicks_pts_2010, density=True, label='Knicks', alpha=0.5)
plt.hist(x=nets_pts_2010, density=True, label='Nets', alpha=0.5)
plt.legend()
plt.title('2010 Season')
plt.show()
plt.clf()

'''
- Answer: The distribution is unimodal for both variables and somewhat overlaping, suggesting there is no strong evidence that the variables has some level of association.
'''

# task 4
knicks_pts_2014 = nba_2014[nba.fran_id == 'Knicks'].pts
nets_pts_2014 = nba_2014.pts[nba.fran_id == 'Nets']

diff_means_2014 = knicks_pts_2014.mean() - nets_pts_2014.mean()
print(diff_means_2014)

plt.hist(knicks_pts_2014, density=True, label='Knicks', alpha=0.5)
plt.hist(nets_pts_2014, density=True, label='Nets', alpha=0.5)
plt.legend()
plt.title('2014 Season')
plt.show()
plt.clf()

'''
- Answer: Mean difference for season 2014 appears minimal between the two teams, evidencing there is no strong association between the variables `fran_id` and `pts`.
'''

# task 5
sns.boxplot(data=nba_2010, x='fran_id', y='pts')
plt.title('Season 2010')
plt.xlabel('Team')
plt.ylabel('Points Scored')
plt.show()
plt.clf()

'''
- Answer: according to the boxplot, overall there are some overlaping between the teams, though the Knicks has the highest top quantiles comparing to the others, and the median points oscilates between 90 to 100 in average. There is no evidence that the variables have strong association from each other.
'''

# task 6
location_result_freq = pd.crosstab(nba_2010.game_location, nba_2010.game_result)
print(location_result_freq)

# task 7
location_result_prop = location_result_freq / len(nba_2010)
print(location_result_prop)

'''
- Answer: The table of contingency shows that the are higher probability to Win playing at `Home`, and less probability to win playing `Away`. It seems that the match location and result has some level of association.
'''

# task 8
print('Table of Frequency:\n', location_result_freq)
chi2, pval, dof, expected = chi2_contingency(location_result_freq)
print('Expected TF:\n', expected)
print('chi2:', chi2)

print('It seems to be some level of association between the two variables (game_location and game_result)')

'''
- Answer:   So there's a clear (although not huge) difference between each pair of numbers (eg. 133 vs 119 or 106 vs. 92). The chi2 value is 6.5 (higher than 4), suggesting that the variables `location` and `result` are  associated.

    - Note: For 2x2 contingency table, Chi-square greated than 4 indicates an association between variables. Conversly otherwise.
'''

# task 9
points_diff_forecast_cov = np.cov(nba_2010.forecast, nba_2010.point_diff)
print(points_diff_forecast_cov)
print('Covariance value:', points_diff_forecast_cov[0][1])

# task 10
point_diff_forecast_corr = pearsonr(nba_2010.forecast, nba_2010.point_diff)
print(point_diff_forecast_corr)

'''
- Answer: The pearson correlation is close to 0.44 positive, which means there is exist a slightly correlation between the two variables.
'''

# task 11
plt.scatter(nba_2010.forecast, nba_2010.point_diff)
plt.title('Forecast - Point Difference Correlation')
plt.xlabel('Forecast WIN Probability')
plt.ylabel('Point Differential')
plt.show()
plt.close()

'''
- Answer: There is a slightly correlation between the varibles, the scatter plot shows that a `higher` 'forecast of winning' tends to have `higher` 'point differential', and conversly a `lower` 'forecast to win' has a `lower` 'point differential'.
'''