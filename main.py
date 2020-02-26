project_path = '/Users/decandia/Dropbox/teresa_stuff/travel_audience' #make sure this points to correct location
model_name = 'model1' #specify the name of the model and the .txt file containing model performance, which will be saved in the path
model_type = 'random_forest' #either 'random_forest' or 'neural_net'. the latter is slow.


#   load packages and set directory
import numpy as np
import os
import pandas as pd
import pickle

os.chdir(project_path)


#   load local packages
from iataDistCalculator import calculate_iata_distance
from varTypes import X_names, y_name
import random_forest
import neural_net
import misc_functions as mf


#   load data that we will analyze
dat = pd.read_csv(os.path.join(project_path, "events_1_1.csv"),
                  parse_dates=['ts', 'date_from', 'date_to'])


#   compute a few features hypothesized to be relevant
dat['duration_trip'] = (dat['date_to'] - dat['date_from']).dt.days #length of trip
dat['time_to_trip'] = (dat['date_from'] - dat['ts']).dt.days #time between timestamp and start of travel
dat['ts_month'] = dat['ts'].dt.month #not so important here since we have only two weeks of data from second half of April 2017
dat['ts_dayofmonth'] = dat['ts'].dt.day #day of month (people may tend to make purchases early/late in month)
dat['ts_weekday'] = dat['ts'].dt.dayofweek.astype(str) #pull day of week from timestamp and treat as category, not numerical
dat['ts_hour'] = dat['ts'].dt.hour

#previous number of searches and bookings for this user - vectorized for speed (apply function was too slow)
#if we had a longer timespan of data we could look at searches/bookings for similar trips (e.g., similar origin and/or destination, or other characteristics of trip)
dat_user_merge = dat[['user_id', 'ts']].merge(dat[['user_id', 'ts', 'event_type']], how = 'inner', on = 'user_id')
dat_user_merge['prev_search'] = ((dat_user_merge['ts_x'] > dat_user_merge['ts_y']) & (dat_user_merge['event_type'] == 'search')) * 1
dat_user_merge['prev_book'] = ((dat_user_merge['ts_x'] > dat_user_merge['ts_y']) & (dat_user_merge['event_type'] == 'book')) * 1
previous_vars = dat_user_merge.groupby(['user_id', 'ts_x']).sum()
dat = dat.merge(previous_vars, how = 'left', left_on = ['user_id', 'ts'], right_on = ['user_id', 'ts_x'])


#   trip distance in km
dat['distance_km'] = calculate_iata_distance(dat['origin'].values, dat['destination'].values) #in order to speed up calculation: a) vectorized function, b) takes np arrays as input


#   convert dependent variable to boolean
dat['book'] = dat['event_type'].map({'book' : True, 'search' : False}) 


#   convert raw data (quality control)
dat.drop_duplicates(inplace=True) #for now let's just get rid of duplicate rows
dat.dropna(inplace=True) #in the future we should impute missing values in some way rather than throw these out (particularly because we want to be able to make predictions for cases with missing values).


#   specify which subset of variables to pull into X and y.
X = dat[X_names]
y = dat[y_name]

#output some descriptive stats for X and y
print("\n"*3, sep = "",
      file=open(os.path.join(project_path, model_name + ".txt"), "w")) #'w' instantiates/rewrites file
print("DESCRIPTIVE STATS\n", sep = "",
      file=open(os.path.join(project_path, model_name + ".txt"), "a"))
print("X:\n", X.describe(include='all'), "\n", sep = "",
      file=open(os.path.join(project_path, model_name + ".txt"), "a"))
print(X.dtypes, "\n", sep = "",
      file=open(os.path.join(project_path, model_name + ".txt"), "a"))
print("y:\n", y.describe(include='all'), "\n", sep = "",
      file=open(os.path.join(project_path, model_name + ".txt"), "a"))
print(y.dtypes, "\n", sep = "",
      file=open(os.path.join(project_path, model_name + ".txt"), "a"))

#Ideally I would probably perform the above QC in the following way:
#Create an instance/member of a class that runs QC on our data
#Feed in the dataset as well as any instructions (e.g., a dictionary of features and their data types) for how I want it processed that are different from the default given by the class attributes
#Instance methods will set X and y as instance attributes

#QC methods would:
#ensure that all columns belonged to one of several data types (e.g., dictionary of requested data types could be fed as argument) and transform columns accordingly (e.g., make sure day of week is treated as category, not numeric)
#run specific qc on each column according to its data type (e.g., formatting dates, creating a category for missing categorical values, dealing with rare categorical values by creating a level for them, etc.)
#.. for each numeric feature, in addition to imputing non-numerical values, we can add a categorical feature that is it's counterpart and tells us whether that feature was missing, NaN, etc.)
#run specific qc according to our knowledge of features, e.g., omit non-sensical values, such as a time_to_trip that is negative (booking occurs after trip begins), or an unusual trip distance (|origin - destination| < 50km)


#   randomly split data into training and test sets
test_proportion = 0.3 #70% training and 30% test
seedInt = 87 #seed used by random number generator to divide data into training and test sets. This should be set to None in future, but for replicability during testing I am setting a seed.

from sklearn.model_selection import train_test_split #import train_test_split function

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_proportion, shuffle=True, random_state=seedInt)

#NOTES on the "use" dataset (the data on which we want to make predictions)
#our QC class instance for model building should preserve/save the relevant QC characteristics so that we can use those for QC on the "use" datasets
#for example, what should happen when the "use" set contains a new categorical level that wasn't in the training set?
#in this cases we might subsume those new categorical levels into the rare level
#for example, what should happen when the "use" dataset contains numerical values that are very different from those in the training set?
#in this case we might set these values to their closest counterparts in the training set
#in each of these cases our QC methods should issue warnings so that we can examine these cases carefully


#   train model
# using random forest classifier, neural net, logistic regression..
# only random forest and neural net implemented so far
if (model_type == 'random_forest'):
    model, y_prob = random_forest.main(X_train, y_train, X_test)
elif (model_type == 'neural_net'):
    model, y_prob = neural_net.main(X_train, y_train, X_test, y_test)
else:
    raise Exception("Set model_type to either 'random_forest' or 'neural_net' and rerun.")


#   evaluate performance
threshold = .2 #proportion with highest likelihood of outcome (these may be the instances when targeted ads will be deployed)
mf.evaluate_model(outcome = np.array(y_test),
                  predicted_prob = y_prob[:,1],
                  top_threshold=threshold,
                  output_path_minus_extension=os.path.join(project_path, model_name))


#   calibrate model
#IF FOR SOME REASON we need the model's predicted probabilities to be accurate, we should not simply assume this is the case.
#Rather we should use binning/grouping to check this and transform them as necessary.
#E.g., if observations with a likelihood of .2 on average have the outcome 30% of the time, then it seems that we are underestimating the probability of the outcome for that group.


#   save model
pickle.dump([model, X_train, X_test, y_train, y_test],
            open(os.path.join(project_path, model_name + ".sav"), 'wb'))
print("\nEnd of output.",
      file=open(os.path.join(project_path, model_name + ".txt"), "a"))
# load the model from disk
#model, X_train, X_test, y_train, y_test = pickle.load(open(os.path.join(project_path, model_name + ".sav"), 'rb'))


'''
scratch:
dat_user_merge = dat[['user_id', 'ts']].merge(dat[['user_id', 'ts', 'event_type']], how = 'inner', on = 'user_id')
previous_vars = dat_user_merge.groupby(['user_id', 'ts_x']).apply(lambda row: pd.Series(dict(
    prev_search=((row['ts_x'] > row['ts_y']) & (row['event_type'] == 'search')).sum(),
    prev_book=((row['ts_x'] > row['ts_y']) & (row['event_type'] == 'book')).sum()
))) #super slow to compute this right now (takes ~1 m). should be the only really slow piece here.
dat = dat.merge(previous_vars, how = 'left', left_on = ['user_id', 'ts'], right_on = ['user_id', 'ts_x'])

'''

