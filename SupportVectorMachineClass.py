# Create our own Support Vector Machine SVM Class
# as compared to Scikit-Learn's SVM
#


import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

####
# build our SVM class. self allows us to use variable
# or functions within the def, that's why we want it
# to be an object which will be reuseable
#
# The __init__ function get's called only once when we create
# create a new instance of the SVM class

class Support_Vector_Machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1:'r', -1:'b'}           # only if we're visualizing
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1) # 1x1 grid, plot #1

# train / optimize our data
#
    def fit(self, data): 
        self.data = data

# our dictionary: magnitude of W ||W|| is our key, our values are just W and b
# { ||W|| : (w,b) }
        opt_dict = {}
    
# we'll apply these transoforms to the vector of W
#
        transforms = [[1,1],
                      [-1,1],
                      [-1,-1],
                      [1,-1]]

        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)
                    
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)  
        all_data = None   # need to re-init each call to our class functions

# start with big steps by max feature value; Q: could we thread these steps?
# Note: to get a more precise value, add more steps here:
#
# what's the logic to decide how small of steps we should take?
# support vectors = yi * (xi dot w + b) = 1
# We'll know we have an optimum w and b when we have a value
# of the above eqn is close to 1; eg stop at 1.01
# Q: can we run these three steps in parallel?
# A: No, since we need the prior knowledge of previous
# larger steps sizes to maintain accuracy
#
        step_sizes = [self.max_feature_value * 0.1,  
                      self.max_feature_value * 0.01,
                      self.max_feature_value * 0.001,]   # point of expense

# now define more init vals; 5 is expensive, this cost 
# is extremely expensive; 
# b doesn't need to take as small as steps as W
# def to 5
        b_range_multiple = 5

# we don't need to take as small of steps with b
# as we do with W
# def to 5
        b_multiple = 5
    
# this is our first element in vector W, first major corner 
# were going to cut
#
        latest_optimum = self.max_feature_value * 10
    
# now we're ready to begin the actual stepping process
# we're at the top of the bowl stepping down to bottom
#
        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum]) # major corner being cut here

# will stay false until we have no more steps to take
# we can do this because of covex optimization
            optimized = False  
            
            while not optimized:   # we want to maximize b here
# arange let's us define of loop step size
# Note: this code black CAN be threaded OK
# we can also optimize the step sizes in b
#
                for b in np.arange(-1 * (self.max_feature_value * b_range_multiple),
                                   self.max_feature_value * b_range_multiple, 
                                   step * b_multiple):

                    for transformation in transforms: # we're going to transform W now
                        w_t = w * transformation
                        found_option = True   ### add a break here later
                        
# weakest link in the SVM fundamentally since we
# need to run this for loop on all the data to check for fit
# SMO attempts to fix this a bit
# Our constraint below needs to be: yi * ((xi dot W) + b) >= 1
#
                        for i in self.data: 
                            for xi in self.data[i]:
                                yi = i
                                if not yi * (np.dot(w_t, xi) + b) >= 1:
                                    found_option = False # we can stop once we find a False

                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b] #magnitude of W
                                
                if (w[0] < 0):
                    optimized = True #we're "step optimized" diff from overall optimized
                    print("Optimized a step")
                else:
# as an example here, w = [5,5]
# step = 1
# w - [step,step] = [4,4]
                    w = w - step   # step some more to find minimum of W

# break out of the entire while loop, then take the next step
# we've hit our optimization value
# here we sort a list of those magnitudes (low to high) 
# remember: ||w|| : [w, b]
#
            norms = sorted([n for n in opt_dict]) # sorted list of all the magnitudes
            opt_choice = opt_dict[norms[0]]  # sorts with minimum at [0]

            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + step * 2 # could match 2 to 10 above

        for i in self.data: 
            for xi in self.data[i]:
                yi = i
                print(xi, ':', yi*(np.dot(self.w, xi) + self.b))

### END of fit function                            
                            
# objective when we visualize the data points is to:
# a. plot our input data points
# b. any predictions we've made
# c. the support vectore hyperplanes
# d. and also the decision boundary hyperplane
# Note: the visualize() function has no bearing on the results, 
# it's only to show the data points, and boundaries
#
    def visualize(self):
        [[self.ax.scatter(x[0] ,x[1], s=100, color=self.colors[i]) for x in data_dict[i]]
                            for i in data_dict]

# defn of a hyperplane is simply: X dot W + b
# what we want is (x dot w + b)
# psv = positive support vector = 1  
# nsv = negative support vector = -1 
# decision boundary = 0
# we want to stipulate what v is equal to
# def hyperplace is purely for us to see the hyperplane, not needed for SVM   
# since all we really need is w and b and the classification result
#                            
        def hyperplane(x,w,b,v):   
            return(-w[0] * x - b + v) / w[1]
                            
# to limit our graph:
        data_range = (self.min_feature_value * 0.9, self.max_feature_value * 1.1) 

        hyp_x_min = data_range[0]
        hyp_x_max = data_range[1]
                            
# now we can create the positive hyperplane
# again: (w dot x + b) = 1
# psv = positive support vector hyperplane:
#
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1) # returns a scalar value
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1) # returns a scalar value                    
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1,psv2], 'k')                    
                            
# now we can create the negative hyperplane
# again: (w dot x + b) = -1
# nsv = negative support vector hyperplane:
#
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1) # returns a scalar value
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1) # returns a scalar value                    
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1,nsv2], 'k') 
                            
# now we can create the decision boundary
# again: (w dot x + b) = 0
# db = decision boundary:
#
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0) # returns a scalar value
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0) # returns a scalar value                    
        self.ax.plot([hyp_x_min, hyp_x_max], [db1,db2], 'y--') 
                            
        plt.show()

### END of visualize function                            

    def predict(self, features):   # prediction is: sign[x dot w + b]
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)

#        print("Classification: ", classification)
#        print("SelfColors: ", self.colors[classification])
#        print("features[0]: ", features[0], "features[1] :", features[1])
        if ((classification != 0) and self.visualization):  # asked for graphing
            self.ax.scatter(features[0], features[1], 
                            s=200, c=self.colors[classification], marker='*')

        return classification
# scatter(self, x, y, s, c, marker, 
# cmap, norm, vmin, vmax, alpha, linewidths, verts, edgecolors, **kwargs)

### END of predict function


data_dict = {-1:np.array([[1,7],
                          [2,8],
                          [3,8],]),
             1:np.array([[5,1],
                         [6,-1],
                         [7,3],]) }

# MAIN: create / init an instance of our Support_Vector_Machine class
#                           
svm = Support_Vector_Machine()     # like: clf
svm.fit(data = data_dict)   
predict_us = [[0,10],
              [1,3],
              [3,4],
              [3,5],
              [5,5],
              [5,6],
              [6,-5],
              [5,8]]

print("Start of predictions:")

for p in predict_us:
    svm.predict(p)
    
print("End of predictions:")

svm.visualize()
                   
##################  End ################
                
# Here's our sample text output:
"""
Output:

Optimized a step
Optimized a step
Optimized a step
Start of predictions:
End of predictions:
"""

