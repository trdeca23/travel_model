# INSTRUCTIONS:


1. Create a working directory for the project with the name of your choice and clone GitHug repository https://github.com/trdeca23/travel_model to it.


2. Install required libraries to your environment. Required libraries are:
  * os
  * pandas
  * sklearn
  * matplotlib
  * scikitplot (currently commented out so you don't technically need this)
  * csv
  * numpy
  * keras
  * seaborn
  * pickle


3. Add the data to this working directory:
  * `events_1_1.csv`
  * `iata_1_1.csv`


4. Make sure that correct paths and names are specified for:
  * `project_path`, specified at the top of main.py script, should point to your project's working directory
  * `model_name`, specified at the top of the main.py script will be the name of the output files created by the script
  * `model_type`, specified at the top of the main.py script should be set to either `'random_forest'` or `'neural_net'`. note that the latter is slow (took ~5 minutes on my laptop). parameters for these models can currently only be set in their respective `.py` files)
  * `iata_path`, specified at the top of the `iataDistCalculator.py`script, should point to `iata_1_1.csv`


5. Run `main.py`. This will call the other .py scripts to clean the data, build the model, and evaluate performance. The following output files will be written to your working directory, using the `model_name` you specify in the step above:
  * `.txt` file with descriptive statistics on features X and target y, and performance metrics for the model. the latter is computed only from the test set
  * `.png` file with ROC plot
  * `.sav` file of model and data that can be reloaded (pickle)
  * `.png` file with importances of features, if you ran `model_type = 'random_forest'`. direction of effect not currently reported.
You may want to change the `model_name` in order to run this several times and keep old output. 





# NOTES: 

- Ran this on my Mac ..I don't think I've used any os specific protocols.

- I wasn't particularly careful with handling datatypes. For example, the training and test data are created as Pandas objects. It may have been better to create these as numpy arrays.

- Stdout and stderr output should ideally have been redirected using `sys.stdout` and `sys.stderr`, for example:
```python 
   import sys
prev_out = sys.stdout
std_log = open('out.log', 'w')
error_log = open('err.log', 'w')
sys.stdout = std_log
print('This message will be logged')
raise Exception('This error will be logged')
sys.stdout = prev_out
std_log.close()
```





# DATA DETAILS:

- `events.csv.gz` - A sample of events collected from an online travel agency, containing:
  * `ts` - the timestamp of the event
  * `event_type` - either `search` for searches made on the site, or `book` for a conversion, e.g. the user books the flight
  * `user_id` - unique identifier of a user
  * `date_from` - desired start date of the journey
  * `date_to` - desired end date of the journey
  * `origin` - IATA airport code of the origin airport
  * `destination` - IATA airport code of the destination airport
  * `num_adults` - number of adults
  * `num_children` - number of children

- `iata.csv` - containing geo-coordinates of major airports
  * `iata_code` - IATA code of the airport
  * `lat` - latitude in floating point format
  * `lon` - longitude in floating point format




