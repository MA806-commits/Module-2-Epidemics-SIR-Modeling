#%% 
# Exploratory Analysis of Mystery Virus Outbreak 

# Right below, we are importing neccessary libraries for data manipulation and visualization. 
    # Pandas (known as variable pd) is used for data manipulation; Matplotlib (known as plt variable) is used for plotting
import pandas as pd
import matplotlib.pyplot as plt

#%%
# Load the data -- Reading file name Module-2-Epidemics-SIR-Modeling/Data/mystery_virus_daily_active_counts_RELEASE#1.csv into a pandas dataframe. We are parsing the 'date' column as dates, and setting header to 0 to indicate that the first row contains column names. index_col=None means we are not setting any column as the index of the dataframe. 
    # Parsing the date column was necessary becasue it allows us to work with the date information better, especially since without parsing, the date would just be treated as a string. 
    # By parsing the date, pandas recognizes the data value as a datetime object, which allows us to perfom date-based operations, such as plotting the data over time, which is what we are needing to do in this analysis: End goal of Day vs. Active Infections plot. 

# Based on the error that I said earlier: 
    # this is the path to the file on my computer: 'C:/Users/kidus/OneDrive/Desktop/Computational BME/Module 02/Module-2-Epidemics-SIR-Modeling/Data/mystery_virus_daily_active_counts_RELEASE#1.csv'
    # This is how the path is saved within Makayla's original start of the code: 'Module-2-Epidemics-SIR-Modeling/Data/mystery_virus_daily_active_counts_RELEASE#1.csv'

data = pd.read_csv('C:/Users/kidus/OneDrive/Desktop/Computational BME/Module 02/Module-2-Epidemics-SIR-Modeling/Data/mystery_virus_daily_active_counts_RELEASE#1.csv', parse_dates=['date'], header=0, index_col=None)

# Makayla's original code for loading the data:
# data = pd.read_csv('Module-2-Epidemics-SIR-Modeling/Data/mystery_virus_daily_active_counts_RELEASE#1.csv', parse_dates=['date'], header=0, index_col=None)

# just a note for error that needs to be addressed: 
    # Kidus -- I encountered an error when trying to load the data and realized that the path to the file is inputed incorreclty within my computer. Locally, within my computer, it is located on my one drive folder, but in order to not cause any technical issues, I am going to address the issue later on after break. 
    # I will to change the path to the file to match where it is located on my computer for the future, but for now, I am just working alongside Makayla's code. 
    # As for contibution on the second day of the project, I will be working from my file path temporarily, and then I will change it to match file paths of both Makayla and I after break.

#%%
# Make a plot of the active cases over time
# Labeling the axes and adding the title to the plot 
plt.xlabel('Day')
plt.ylabel('Active Cases')
plt.title('Active Cases of Mystery Virus Over Time')

#Loading the data into the plot. 
    # We are plotting the 'day' column on the x-axis and the 'active reported daily cases' column on the y-axis. We are using marker='o' to indicate that we want to use circles to mark each data point, and linestyle='-' to connect the data points with a line.
plt.plot(data['day'], data['active reported daily cases'], marker='o', linestyle='-')

plt.show() # -- This line is used to display the plot. 


## Response to Questions:  -- Answered by Makayla
    # 1. What do you notice about the initial infections?: 
        # The initial phase (Days 0-20) shows exponential growth. At the beginning, the number of active cases stays very low for nearly 10 days, making the threat appear decivingly minimal. Despite the low numbers, the looks to be doubling at a constant rate. Around Day 20, the curve bends sharply upwards, representing a quick rate of increase of active cases. 

    # 2. How could we measure how quickly its spreading?: 
        # We could measure the growth rate (r) of the exponential curve during the initial phase. This growth rate can be used to estimate the basic reproduction number (R0) of the virus, which indicates how many people, on average, one infected person will infect in a fully susceptible population. 
        # By looking at the time between the first exposure and the first sharp spike, you can estimate the virus's incubation period. 
        
    # 3. What information about the virus would be helpful in determining the shape of the outbreak curve?: 
        # Information about the virus's incubation period, infectious period, and transmission mode would be helpful. 
        # Also, data on population density, social behavior, and intervention measures (like social distancing) would also influence the shape of the curve, indireclty impacting the rate of spread, peak number of active cases, and rate of declining cases associated with the specific virus.