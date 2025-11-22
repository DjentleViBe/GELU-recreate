import csv 

def csv_write(filename, x, predicted, xname, yname, zname, y_actual):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([xname, yname, zname])  # header
        for xi, yi, zi in zip(x, predicted, y_actual):
            writer.writerow([xi.item(), yi.item(), zi.item()])  # convert tensors to Python floats


def csv_write2(filename, x, predicted, xname, yname, zname, wname, y_actual, w_actual):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([xname, yname, zname, wname])  # header
        for xi, yi, zi, wi in zip(x, predicted, y_actual, w_actual):
            writer.writerow([xi.item(), yi.item(), zi.item(), wi.item()]) 

def csv_read(filename, xname, yname, zname):
    x_vals = []
    y_preds = []
    z_preds = []
    reader = csv.DictReader(filename)  # read as dictionary
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x_vals.append(float(row[xname]))
            y_preds.append(float(row[yname]))
            z_preds.append(float(row[zname]))
    return x_vals, y_preds, z_preds


def csv_read2(filename, xname, yname, zname, wname):
    x_vals = []
    y_preds = []
    z_preds = []
    w_preds = []
    reader = csv.DictReader(filename)  # read as dictionary
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x_vals.append(float(row[xname]))
            y_preds.append(float(row[yname]))
            z_preds.append(float(row[zname]))
            w_preds.append(float(row[wname]))
    return x_vals, y_preds, z_preds, w_preds
