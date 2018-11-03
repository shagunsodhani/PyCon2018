
# coding: utf-8

# # [Numpy To PyTorch](https://colab.research.google.com/drive/1zEFCe68fJvdJHwTo8X-dCHunxNDAIFep)
# 
# 
# 

# In[ ]:


import numpy as np
import torch
import time


# ## Numpy
# 

# #### Numpy vs Python List

# In[4]:


given_list = [1, 2, 3]
new_array = np.array(given_list)

print(type(new_array))

# <class 'numpy.ndarray'>


# In[5]:


given_list = [24, 12, 57]
new_array = np.array(given_list)

print([x+3 for x in given_list])

print(new_array+3)


# In[6]:


first_array = np.random.rand(128, 5)

second_array = np.random.rand(5, 128)

print(np.matmul(first_array, second_array))


# ## PyTorch

# In[7]:


torch.Tensor


# #### Creation Ops

# In[8]:


shape = (2, 3)
print(np.ones(shape))
print(torch.ones(shape))


# In[9]:


shape = (3, 3)
x = 2 * np.ones(shape)
y = np.eye(shape[0])
print(x+y)

x = 2 * torch.ones(shape)
y = torch.eye(shape[0])
print(x+y)


# In[10]:


x = np.array([[1, 2], [3, 4]])	
print(x)

x = torch.tensor([[1, 2], [3, 4]])
print(x)


# #### Variable Assignment

# Numpy supports two styles:
# 
# * Compute and assign to a variable using the `assignment` operator.
# * Compute and assign value using a function call

# In[11]:


shape = (3, 3)
x = 2 * np.ones(shape)
y = np.eye(shape[0])
y = np.add(x, y)
print(y)



x = 2 * np.ones(shape)
y = np.eye(shape[0])
np.add(x, y, out=y)
print(y)


# In[12]:


shape = (3, 3)
x = 2 * torch.ones(shape)
y = torch.eye(shape[0])
y = torch.add(x, y)
print(y)



x = 2 * torch.ones(shape)
y = torch.eye(shape[0])
torch.add(x, y, out=y)
print(y)


# #### Advanced Indexing

# In[13]:


x = np.arange(10)
print(x[1:7:2])


y = np.arange(35).reshape(5,7)
print(y[1:5:2,::3])


# In[14]:


x = torch.arange(10)
print(x[1:7:2])

y = torch.arange(35).reshape(5,7)
print(y[1:5:2,::3])


# In[15]:


print(np.random.random(shape))
torch.rand(shape)


# In[16]:


x = np.arange(10,1,-1)
indexing_array = np.array([3,3,-3,8])
print(x[indexing_array])

indexing_array = np.array([[1,1],[2,3]])
print(x[indexing_array])


# In[17]:


x = torch.arange(10,1,-1)
indexing_array = torch.tensor([3,3,-3,8])
print(x[indexing_array])


indexing_array = torch.tensor([[1,1],[2,3]])
print(x[indexing_array])


# #### Other comparisons for the APIs

# In[18]:


shape = (2000, 1000)

np_array = np.ones(shape)
np.sum(np_array, axis=1)


# In[19]:


torch_array = torch.ones(shape)
torch.sum(torch_array, dim=1)


# In[20]:


x = np.linspace(start=10.0, stop=20, num=5)
print(x)

x = torch.linspace(start=10, end=20, steps=5)
print(x)


# #### GPU Acceleration

# In[21]:


get_ipython().run_cell_magic('timeit', '', 'np.random.seed(1)\nn = 10000\nx = np.array(np.random.randn(n,n), dtype = np.float32)\ny = np.matmul(x, x)')


# In[22]:


get_ipython().run_cell_magic('timeit', '', "torch.manual_seed(1)\nn = 10000\ndevice = torch.device('cuda:0')\nx = torch.rand(n, n, dtype=torch.float32, device=device)\ny = torch.matmul(x, x)")


# In[23]:


get_ipython().run_cell_magic('timeit', '', 'torch.manual_seed(1)\nn = 10000\nx = torch.rand(n, n, dtype=torch.float32)\ny = torch.matmul(x, x)')


# #### PyTorch to Numpy to PyTorch

# In[24]:


shape = (5, 3)
numpy_array = np.array(shape)
torch_array = torch.from_numpy(numpy_array)
print(torch_array)
recreated_numpy_array = torch_array.numpy()
print(recreated_numpy_array)
if((recreated_numpy_array == numpy_array).all()):
  print("Numpy -> Torch -> Numpy")


# #### Tensors to GPU

# In[ ]:


tensor = torch_array


# In[ ]:


gpu_device = torch.device('cuda:0')
cpu_device = torch.device('cpu')
tensor_on_gpu = tensor.to(gpu_device)
tensor_on_cpu = tensor.to(cpu_device)


# ## Pitfalls

# #### Pitfall1

# In[27]:


numpy_array = np.array([1, 2, 3])
torch_array = torch.from_numpy(numpy_array)
torch_array[0] = -100
print(numpy_array[0])


# In[28]:


numpy_array = np.array([1, 2, 3])
torch_array = torch.Tensor(numpy_array)
torch_array[0] = -100
print(numpy_array[0])


# #### Pitfall2

# In[29]:


numpy_array = np.array([1, 2, 3])
torch_array = torch.from_numpy(numpy_array).to(gpu_device)
torch_array[0] = -100
print(numpy_array[0])


# #### Pitfall3

# In[30]:


torch_array = torch.tensor([1, 2, 3], 
            device = gpu_device)
numpy_array = torch_array.numpy()


# In[31]:


torch_array = torch.tensor([1, 2, 3], 
                device = gpu_device)

numpy_array = torch_array
            .to(cpu_device)
            .numpy()


# In[32]:


shape = (128, 1000)
np_array = np.random.random(shape)
np.sum(np_array, axis=1)


# In[33]:


shape = (128, 1000)
torch_array = torch.rand(shape)
torch.sum(torch_array, dim=1)


# ## More Indexing Examples

# In[34]:


x = np.arange(10)
print(x[2:5])
print(x[:-7])
print(x[1:7:2])
y = np.arange(35).reshape(5,7)
print(y)
print(y[1:5:2,::3])


# In[35]:


x = torch.arange(10)
print(x[2:5])
print(x[:-7])
print(x[1:7:2])
y = torch.arange(35).reshape(5,7)
print(y)
print(y[1:5:2,::3])


# In[36]:


x = np.arange(10,1,-1)
print(x)
indexing_array = np.array([3, 3, 1, 8])
print(x[indexing_array])
indexing_array = [np.array([3,3,-3,8])]
print(x[indexing_array])
indexing_array = np.array([[1,1],[2,3]])
print(x[indexing_array])


# In[37]:


x = torch.arange(10,1,-1)
print(x)
indexing_array = torch.tensor([3, 3, 1, 8])
print(x[indexing_array])
indexing_array = [torch.tensor([3,3,-3,8])]
print(x[indexing_array])
indexing_array = torch.tensor([[1,1],[2,3]])
print(x[indexing_array])


# In[38]:


x = np.arange(10,1,-1)
indexing_array = np.array([3,3,-3,8])
print(x[indexing_array])

indexing_array = np.array([[1,1],[2,3]])
print(x[indexing_array])


# In[39]:


x = torch.arange(10,1,-1)
indexing_array = torch.tensor([3,3,-3,8])
print(x[indexing_array])
# [7 7 4 2]


indexing_array = torch.tensor([[1,1],[2,3]])
print(x[indexing_array])
# [[9 9]
# [8 7]]


# In[40]:


shape = (2, 3)
print(np.ones(shape))
print(torch.ones(shape))
# [[1. 1. 1.]
# [1. 1. 1.]]

