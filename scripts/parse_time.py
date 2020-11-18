import sys
import numpy as np

times = list()

with open(sys.argv[1]) as file:
    for line in file:
        if line.startswith("[R]: "):
            time = float(line.split(':')[1])
            times.append(time)
        try:
            ls = line.split(', ')
            print('Sent/s:         {:.2f}'.format(float(ls[0])/float(ls[1])))
        except:
            pass

times.remove(times[0])
times = np.array(times)

print('Mean:           {:.2f}'.format(1000*np.mean(times)))
print('99 Percentile:  {:.2f}'.format(1000*np.percentile(times, 99)))
print('999 Percentile: {:.2f}'.format(1000*np.percentile(times, 99.9)))
print('Total:          {:.2f}'.format(np.sum(times)))
