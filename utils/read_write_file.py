import csv
import fileinput
def write_array_to_file(filename,array):
    with open(filename, 'ab') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        for element in array:
            if type(element) is not list:
                element=[element]
            spamwriter.writerow(element)

def read_file_as_array(filename):
    try:
        output=[]
        with open(filename,'r') as file:
            for line in file:
                output.append(line.rstrip())
        return output
    except Exception as e:
        return []
    