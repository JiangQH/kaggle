import scipy as sp
import os.path as osp
import matplotlib.pyplot as plt
from tools import DATA_DIR

data = sp.genfromtxt(osp.join(DATA_DIR, 'web_traffic.tsv'), delimiter='\t')
print data.shape
print data[:10]

x = data[:, 0]
y = data[:, 1]
print sp.sum(sp.isnan(y))

x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]

def plot(x, y, f=None, title=None):
    colors = ['g', 'k', 'b', 'm', 'r']
    plt.figure(title)
    plt.scatter(x, y)
    plt.xlabel("Web traffic over the last month")
    plt.ylabel("Hits/hour")
    plt.xticks([w * 7 * 24 for w in range(5)],
               ['week {}'.format(w) for w in range(5)])
    if f:
        fx = sp.linspace(0, x[-1], 1000)
        for model, color in zip(f, colors):
            plt.plot(fx, model(fx), linewidth=2, c=color)
            plt.legend(["d=%i" % model.order], loc="upper left")
    plt.show()

#plot(x, y, title='data')

def error(f, x, y):
    return sp.sum((f(x)-y) ** 2)

fp1, residuals, rank, sv, rcond = sp.polyfit(x, y, 3, full=True)
#print "model parameters: {}".format(fp1)
#print "errors: {}".format(residuals)
function1 = sp.poly1d(fp1)
#print "total error : {}".format(error(function, x, y))
plot(x, y, [function1], 'fit line')



inflection = 3.5 * 7 * 24
xa = x[:inflection]
ya = y[:inflection]
xb = x[inflection:]
yb = x[inflection:]
fa = sp.poly1d(sp.polyfit(xa, ya, 1))
fb = sp.poly1d(sp.polyfit(xb, yb, 1))
plot(x, y, [fa, fb], 'fit two line')












