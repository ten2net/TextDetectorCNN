import os
name='fcn'
for _name in os.listdir('../visuals/'):
    if 'txt' not in _name:
        continue
    if 'log_'+name!=_name[:-4]:
        continue
    logs=[i for i in open('../visuals/'+_name).readlines() if 'E' not in i]
    y=[]
    for i in logs:
        y.append([float(j) for j in i.split() if j.replace('.','').isdigit()])
    from matplotlib import pyplot
    x=[i+1 for i in range(len(y))]
    pyplot.plot(x,[i[0] for i in y],linewidth=1.5)
    pyplot.plot(x,[i[1] for i in y],linewidth=1.5)
    pyplot.plot(x,[i[2] for i in y],linewidth=1.5)
    pyplot.plot(x,[i[3] for i in y],linewidth=1.5)
    #pyplot.plot(x,[min([i[0] for i in y][:i+1]) for i in range(len(y))],linewidth=0.25)
    #pyplot.plot(x,[max([i[1] for i in y][:i+1]) for i in range(len(y))],linewidth=0.25)
    #pyplot.plot(x,[min([i[2] for i in y][:i+1]) for i in range(len(y))],linewidth=0.25)
    #pyplot.plot(x,[max([i[3] for i in y][:i+1]) for i in range(len(y))],linewidth=0.25)
    #pyplot.plot(x,[max([i[0] for i in y][i:]) for i in range(len(y))],linewidth=0.25)
    #pyplot.plot(x,[min([i[1] for i in y][i:]) for i in range(len(y))],linewidth=0.25)
    #pyplot.plot(x,[max([i[2] for i in y][i:]) for i in range(len(y))],linewidth=0.25)
    #pyplot.plot(x,[min([i[3] for i in y][i:]) for i in range(len(y))],linewidth=0.25)
    for i in range(len(y)-7):
        for j in range(4):
            for k in range(1,8):
                y[i][j]+=y[i+k][j]
            y[i][j]/=8
    pyplot.plot(x[:-7], [i[0] for i in y][:-7], linewidth=0.25)
    pyplot.plot(x[:-7], [i[1] for i in y][:-7], linewidth=0.25)
    pyplot.plot(x[:-7], [i[2] for i in y][:-7], linewidth=0.25)
    pyplot.plot(x[:-7], [i[3] for i in y][:-7], linewidth=0.25)
    pyplot.legend(['loss', 'acc', 'val_loss', 'val_acc'], loc='best')
    pyplot.savefig('../visuals/'+_name[:-4]+'.jpg')
    pyplot.title(_name[:-4])
    pyplot.show()
    pyplot.clf()