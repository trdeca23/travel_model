iata_file = 'iata_1_1.csv' #make sure this points to correct location

import csv
import math


#Reads in iata_file as dictionary with:
#key name - IATA code,
#value - tuple of ordered floats corresponding to latitude and longitude
iata_dict = {}
with open(iata_file) as csv_file:
	csv_reader = csv.DictReader(csv_file)
	for row in csv_reader:
		iata_dict[row['iata_code']] = (float(row['lat']), float(row['lon']))


# Haversine formula example in Python
# Author: Wayne Dyck
def distance(origin, destination):
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371 # km
    #radius = 3959 # miles
    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c
    return d

#Function whose inputs are:
#start_loc, end_loc - IATA codes corresponding to start and end airports.
#Uses latitude and longitude coordinates located in iata_dict and calculates distance using distance function
#Returns:
#dist - difference between end_loc and start_loc
def calculate_iata_distance(start_loc, end_loc):
	start_coord = iata_dict[start_loc]
	end_coord = iata_dict[end_loc]
	dist = distance(start_coord, end_coord)
	#print(start_coord, end_coord)
	return(dist)
	

#NOTES:
#I'm not sure why the same IATA codes appear multiple times. I am currently using the code that appears last (overwriting earlier rows in iata_file).
