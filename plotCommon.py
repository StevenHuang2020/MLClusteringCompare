import matplotlib.pyplot as plt

def plot(x,y):
    plt.plot(x,y)
    plt.show()

def plotSub(x,y,ax=None, aspect=False, label='',c='k',marker='.'):
    ax.plot(x,y,label=label,c=c,marker=marker)
    #ax.title.set_text(name)
    if aspect:
        ax.set_aspect(1)
    ax.legend()
        
def scatterSub(x,y,ax=None,label='',marker=',',c='r'):
    ax.scatter(x,y,c=c,label=label,marker=marker) #linewidths=.3,
    #ax.set_aspect(1)

def scatter(x,y,ratio=True):
    plt.scatter(x,y)
    if ratio:
        ax = plt.gca()
        ax.set_aspect(1)
    plt.show()