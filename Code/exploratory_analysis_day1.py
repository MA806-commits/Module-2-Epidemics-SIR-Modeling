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

## Response to Questions: 
# What do you notice about the initial infections?: The initial phase (Days 0-20) shows exponential growth. At the beginning, the number of active cases stays very low for nearly 10 days, making the threat appear minimal. Despite the low numbers, the looks to be doubling at a constant rate. Around Day 20, the curve bends sharply upwards. 
# How could we measure how quickly its spreading?: We could measure the growth rate (r) of the exponential curve during the initial phase. This growth rate can be used to estimate the basic reproduction number (R0) of the virus, which indicates how many people, on average, one infected person will infect in a fully susceptible population. By looking at the time between the first exposure and the first sharp spike, you can estimate the virus's incubation period. 
# What information about the virus would be helpful in determining the shape of the outbreak curve?: Information about the virus's incubation period, infectious period, and transmission mode would be helpful. Also, data on population density, social behavior, and intervention measures (like social distancing) would also influence the shape of the curve. 