import csv

input1 = open("maros_history_full.csv", 'r')
output = open("maros_history_full_clean.csv", 'w')
writer = csv.writer(output)
for row in csv.reader(input1):
    if row:
        writer.writerow(row)
input1.close()
output.close()