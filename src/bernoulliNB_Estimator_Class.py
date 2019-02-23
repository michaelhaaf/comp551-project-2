from math import log
import numpy as np

class bernoulliNB_Estimator():
    def __init__(self):
        
        from math import log
        import numpy as np
        
       
        self.W0 =0 #bias
        self.W =[] #weights
        
        
        
        #private
        self.w_j_0 = []
        self.w_j_1 =[]
        self.y_count = 0   #number of instances where y=1 
        self.y = 0
        self.theta_j_0= [] #[P(xj=1|y=0)]
        self.theta_j_1= [] #[P(xj=0|y=0)]
        self.theta_1 =  0 #[P(y=1)]
        self.prediction = []
        self.num_ins = 0   #number of instances in training dataset
        self.num_feat = 0  #number of instances in training dataset
        
        self._c = 0
        self._sum_w_j_0 = 0 
        self.nonzeroelment_row=0
        self.nonzeroelment_column=0
        self.x_train = []
        self.y_train = []
        self.val_instances=0
        self.val_feat=0
        self.prediction=[]
        self.range_sparse=0
        self.row=0
        self.col=0
        
    def fit(self,X_train, Y,features_list):
        self.features=features_list
        
        
        #assuming X_train is sparse
        self.y_train=Y
        self.num_ins=len(self.y_train)
        self.num_feat=X_train.shape[1]
        self.x_train=find(X_train) #tuple
        self.Count_y_train_equals_1 ()
        self.theta_1 = self.y_count/self.num_ins
        self.compute_theta_j()
        
        #compute theta/theta_j_0/theta_j_1
        
         #computes theta_j_0 and theta_j_1
        #compute w_j_0/w_j_1
        self.compute_w_j()
        #compute the weights and the bias 
        self.compute_Weights()
      
        
    
    
    
    def predict(self,X_val):  
        self.val_instances=X_val.shape[0]
        self.val_feat=X_val.shape[1]
        self.x_val=find(X_val) 
        self.prediction=[0]*self.val_instances
        self.range_sparse=len(self.x_val[0]) 
        for i in range (self.range_sparse):
            self.row=self.x_val[0][i]
            self.col=self.x_val[1][i]
            self.prediction[self.row]=self.W[self.col]+self.prediction[self.row]
            print(1)
            print(i)
        
        for i in range (self.val_instances):
            self.prediction[i]=self.prediction[i]+self.W0
            print(2)
            print(i)
            
            if (self.prediction[i]>=0):
                self.prediction[i]=1
            else:
                self.prediction[i]=0
        return self.prediction
    def Count_y_train_equals_1 (self):
        self.y_count=0
        for i in range (self.num_ins):  
            if(self.y_train[i]==1):
                self.y_count=self.y_count+1     
                
                
                
                
                
                
                
                
                
                
    def compute_theta_j(self):
        self.y=0
        self.theta_j_0=np.zeros((self.num_feat,1))
        self.theta_j_1=np.zeros((self.num_feat,1))
        #non sparse X
        #for i in range (self.num_ins): 
            #for j in range (self.num_feat):
                #if (X_train[i][j]==1  and y_train[i]==0):
                    #self.theta_j_0 [j]=self.theta_j_0[j]+1   
                #if (X_train[i][j]==1  and y_train[i]==1):
                    #self.theta_j_1 [j]=self.theta_j_1[j]+1
        #sparse X
        size=len(self.x_train[0])
        for i in range (size): 
            self.nonzeroelment_row=self.x_train [0][i]
            self.nonzeroelment_column=self.x_train[1][i]
    
    
            if (self.y_train[self.nonzeroelment_row]==0): #theta_j_0
                self.theta_j_0[self.nonzeroelment_column]=self.theta_j_0[self.nonzeroelment_column]+1
            else:
              #theta_j_1
                self.theta_j_1[self.nonzeroelment_column]=self.theta_j_1[self.nonzeroelment_column]+1
        
            print(i)
        #laplace smoothing is done here(self.num_ins-self.y_count+2)
        for j in range (self.num_feat):
            self.theta_j_0 [j]=(self.theta_j_0 [j]+1)/(self.num_ins-self.y_count+2)
            self.theta_j_1 [j]=(self.theta_j_1[j]+1)/(self.y_count+2)
            
            
            
    def compute_w_j(self) :
        self.w_j_0=np.zeros((self.num_feat,1))
        self.w_j_1=np.zeros((self.num_feat,1))
        for j in range (self.num_feat):
            self.w_j_0[j][0]=log((1-self.theta_j_1[j][0])/(1-self.theta_j_0[j][0]) )
            self.w_j_1[j][0]=log (self.theta_j_1[j][0]/self.theta_j_0[j][0])   
        
        
    def compute_Weights(self):
        
        self.W1=np.zeros((self.num_feat,1)) 
        self._sum_w_j_0=0
        
        if (self.theta_1!=1):
            self._c= log(self.theta_1/(1-self.theta_1))
        else:
            print('Pr(y=0)=0 ... log (Pr(y=1)/Pr(y=0)) not defined: Invert zeros and ones in output')
            return
        
        self._sum_w_j_0=sum(self.w_j_0)
        
        self.W0=sum(self._sum_w_j_0) + self._c
        
             
            #self.W [j] = self.w_j_1[j] - self.w_j_0[j]
            
        self.W =numpy.subtract(self.w_j_1, self.w_j_0)
            
