import csv

with open(r"C:\Users\yansokolov\Downloads\icite_metadata_5.csv", 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    print(row[1])