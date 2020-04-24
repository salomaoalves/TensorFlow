import numpy as np 
import pandas as pd 
import tensorflow as tf 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

class RLS:
    def __init__(self): #construct, initialize some importants variables
        self.b0 = tf.Variable(0.54)
        self.b1 = tf.Variable(0.72)
        self.b0_final = 0
        self.b1_final = 0
        self.batch_size = 32
        self.xph = tf.placeholder(tf.float32, [self.batch_size, 1])
        self.yph = tf.placeholder(tf.float32, [self.batch_size, 1])
        self.y_model = self.b0 + self.b1 * self.xph
        self.err = tf.losses.mean_squared_error(self.yph, self.y_model)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
        self.training = self.optimizer.minimize(self.err)
        self.init = tf.global_variables_initializer()

    def train(self, x, y):
        with tf.Session() as sess: #start the train
            sess.run(self.init)
            for _ in range(1000):
                indices = np.random.randint(len(X), size = self.batch_size)
                feed = {self.xph: X[indices], self.yph: y[indices]}
                sess.run(self.training, feed_dict = feed) 
            self.b0_final, self.b1_final = sess.run([self.b0, self.b1])

    def pred(self, X): #make predictions
        p = self.b0_final + self.b1_final * X
        return p

    def graph(self,x,y,p): #plot the data and the predictions
        plt.plot(x,y,'o')
        plt.plot(x,p,color='red')
        plt.xlabel("Speed")
        plt.ylabel("Distance")
        plt.show()

    def error(self,y,p): #compute the error
        print("Mean Absolute Error:", mean_absolute_error(y, p))
        print("Mean Squared Error:", mean_squared_error(y, p))


if __name__ == '__main__':

    #load and tranformation the data
    base = pd.read_csv('database/cars.csv',sep=',')
    x = base.iloc[:, 1].values
    x = x.reshape(-1,1)
    y = base.iloc[:, 2].values
    y = y.reshape(-1,1)
 
    #put the data in the same scale
    scaler_x = StandardScaler()
    X = scaler_x.fit_transform(x)
    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(y)

    model = RLS()
    model.train(X,y)
    p = model.pred(X)
    model.graph(X,y,p)
    model.error(y,p)
