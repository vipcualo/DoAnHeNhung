import glob
import numpy as np
import random
def read_data(path,y,dim1,dim2):
    files=glob.glob(path+"/*")
    random.choice(files)
    X=[]
    Y=[]
    for file in files:
        f=open(file,"r")
        x=[]
        count=0
        for line in f:
            count+=1
            if count > 25000 :
                break
            g=line.split(";")[0]
            i=line.split(";")[1]
            x.append((float(g)/127))
            x.append((float(i)/127))
        #print(x)
        for j in range(0,len(x),dim1*dim2*2):
                if (len(x)< (j+dim1*dim2*2)):
                    break
                x1=x[j:(j+dim1*dim2*2)]
                x1=np.reshape(x1,(dim1,dim2,2))
                X.append(x1)
                Y.append(y)
        f.close()
    X=np.array(X)
    Y=np.array(Y)
    return X,Y