
# coding: utf-8

# In[1]:


get_ipython().magic('matplotlib inline')
import pandas as pd              # A beautiful library to help us work with data as tables
import numpy as np               # So we can use number matrices. Both pandas and TensorFlow need it. 
import matplotlib.pyplot as plt  # Visualize the things
import tensorflow as tf          # Fire from the gods


# In[37]:


dataframe = pd.read_csv("bugs_Int.csv") # Let's have Pandas load our dataset as a dataframe
#dataframe = dataframe.drop(["index", "price", "sq_price"], axis=1) # Remove columns we don't care about
#dataframe = dataframe[0:10] # We'll only use the first 10 rows of the dataset in this example
dataframe


# In[38]:


inputX = dataframe.loc[:, ['LOC', 'LLOC','SLOC','Halstead Volume','CC','h','N','calculated_le0gth']].as_matrix()


# In[39]:


inputX


# In[40]:


inputY = dataframe.loc[:, ["Buggy"]].as_matrix()


# In[41]:


inputY


# In[54]:


learning_rate = 0.001
training_epochs = 2000
display_step = 50
n_samples = inputY.size


# In[55]:


print(n_samples)


# In[56]:


x = tf.placeholder(tf.float32, [None, 8])

W = tf.Variable(tf.zeros([8, 1]))
b = tf.Variable(tf.zeros([8]))
y_values = tf.add(tf.matmul(x, W), b)
y = tf.nn.softmax(y_values)
y_ = tf.placeholder(tf.float32, [None,1])


# In[57]:



# Cost function: Mean squared error
cost = tf.reduce_mean(tf.pow(y_ - y, 2)) / 2
#cost = tf.reduce_sum(tf.pow(y_ - y, 2))/(2*n_samples)
# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# In[58]:


init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


# In[59]:


for i in range(training_epochs):  
    sess.run(optimizer, feed_dict={x: inputX, y_: inputY}) # Take a gradient descent step using our inputs and labels

    # That's all! The rest of the cell just outputs debug messages. 
    # Display logs per epoch step
    if (i) % display_step == 0:
        cc = sess.run(cost, feed_dict={x: inputX, y_:inputY})
        print ("Training step:", '%04d' % (i), "cost=", "{:.9f}".format(cc))

print("Optimization Finished!")
training_cost = sess.run(cost, feed_dict={x: inputX, y_: inputY})
print ("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')


# In[34]:


sess.run(y, feed_dict={x: inputX })


# In[36]:


sess.run(tf.nn.softmax([1., 2.]))


# In[ ]:




