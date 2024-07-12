
import os
import pandas as pd


#path = "C2TM_LongForecasting"
# path = "CBS_LongForecasting"
path = "Milano_LongForecasting"

files = os.listdir(path)

files = sorted(files)

# write_clo = ['methods','4-mse', '4-mae', '5-mse', '5-mae','6-mse', '6-mae', '8-mse', '8-mae']

# df = pd.DataFrame(columns=(write_clo))

#df = pd.read_csv('c2tm.csv')


metrics = []

for i, f in enumerate(files):
    f_names = f.split("_")
    file_path = os.path.join(path, f)
    mtx = [f_names[0]+f_names[-1][:-4]]
    if os.path.isfile(file_path):
        with open(file_path, 'r') as f:
            line = f.readline()
            while line:
                if "mse:" in line and "mae:" in line:
                    line = line.replace(' ','')
                    print(line)
                    ctx=line.split(',')
                    mtx.append('%.3f'%(float(ctx[0][4:])))
                    mtx.append('%.3f'%(float(ctx[1][4:])))
                    #mtx.append('%.2f'%float(ctx[2][4:]))
                line = f.readline()
    print(mtx)
    metrics.append(mtx)

print('----------------------------------------------------------------')
#print(metrics)

min_mse = 10000
min_mae = 10000

mse_ok =[[]]
mae_ok = [[]]

for mm in metrics:
    if '72' in mm[0]:
        if float(mm[1]) < min_mse:
            mse_ok[0] = mm
            min_mse = float(mm[1])
        if float(mm[2]) < min_mae:
            mae_ok[0] = mm
            min_mae = float(mm[2])
print(min_mse,' = min_mse = ', mse_ok)
print(min_mae,' = min_mae = ', mae_ok)



