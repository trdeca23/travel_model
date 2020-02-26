iata_path = 'iata_1_1.csv' #make sure this points to correct location

import csv
import numpy as np


#Reads in iata_path as dictionary with:
#key name - IATA code,
#value - tuple of ordered floats corresponding to latitude and longitude
iata_dict = {}
with open(iata_path) as csv_file:
	csv_reader = csv.DictReader(csv_file)
	for row in csv_reader:
		iata_dict[row['iata_code']] = (float(row['lat']), float(row['lon']))


# Haversine formula
#https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
def haversine(lat1, lon1, lat2, lon2):
    KMS = 6371 #radius of earth in km
    MILES = 3959 #radius in miles
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1 
    dlon = lon2 - lon1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    total_kms = KMS * c
    return total_kms


#Function whose inputs are:
#start_loc, end_loc - pandas series of IATA codes corresponding to start and end airports.
#Uses latitude and longitude coordinates located in iata_dict and calculates distance using distance function
#Returns:
#dist - difference between end_loc and start_loc
def calculate_iata_distance(start_loc, end_loc):
        lat1, lon1 = zip(*[iata_dict[x] for x in start_loc])
        lat2, lon2 = zip(*[iata_dict[x] for x in end_loc])
        #lat1, lon1 = zip(*start_loc.apply(lambda x: iata_dict[x])) faster if input is numpy array instead of pandas series
        #lat2, lon2 = zip(*end_loc.apply(lambda x: iata_dict[x]))
        dist = haversine(lat1, lon1, lat2, lon2)
        return(dist)



#NOTES:
#I'm not sure why the same IATA codes appear multiple times. I am currently using the code that appears last (overwriting earlier rows in iata_path).
