#%%
import pandas as pd
import matplotlib.pyplot as plt

#%%
# Load the data
data = pd.read_csv('Module-2-Epidemics-SIR-Modeling/Data/mystery_virus_daily_active_counts_RELEASE#1.csv', parse_dates=['date'], header=0, index_col=None)

#%%
# Make a plot of the active cases over time
# Labeling 
plt.xlabel('Day')
plt.ylabel('Active Cases')
plt.title('Active Cases of Mystery Virus Over Time')

#Loading the data
plt.plot(data['day'], data['active reported daily cases'], marker='o', linestyle='-')

plt.show()