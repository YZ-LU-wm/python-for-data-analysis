# 4.1 NumPy ndarray


```python
import numpy as np #导入numpy包
```


```python
data=np.random.randn(2,3) #使用函数randn()产生一个数组，并赋值给变量data
```


```python
data #输出data
```




    array([[-0.25607645,  0.13587343,  0.61622153],
           [ 0.86006928, -0.93994827,  0.64518628]])




```python
data*10 #对数组中每个元素做乘法
```




    array([[-2.56076451,  1.35873433,  6.16221534],
           [ 8.6006928 , -9.39948272,  6.45186278]])




```python
data+data #素组相加
```




    array([[-0.5121529 ,  0.27174687,  1.23244307],
           [ 1.72013856, -1.87989654,  1.29037256]])



## 4.1.1 ndarray属性

1、属性`arrayinstance.shape`返回数组每一位的维数。


```python
data.shape #输出数组每一维的维数
```




    (2, 3)



2、属性`arrayinstance.dtype`返回数组的数据类型。


```python
data.dtype #输出数组的数据类型
```




    dtype('float64')



3、属性`arrayinstance.T`返回数组arrayinstance的转置。


```python
data.T
```




    array([[-0.25607645,  0.86006928],
           [ 0.13587343, -0.93994827],
           [ 0.61622153,  0.64518628]])



4、属性`arrayinstance`返回数组的元素个数。即，如果数组是2*3维的，那么size将返回6。


```python
data.size
```




    6



5、属性`arrayinstance`返回数组的第一维维数。


```python
data.ndim
```




    2



## 4.1.2 生成ndarray


```python
#1、使用函数array()生成数组。函数array()接收列表,并以列表数据作为参数，按照指定格式返回一个n维数组。
data1=[6,7.5,8,0,1]
```


```python
arr1=np.array(data1,ndmin=1)
```


```python
arr1
```




    array([6. , 7.5, 8. , 0. , 1. ])




```python
data2=[[1,2,3,4],[5,6,7,8]]
```


```python
arr2=np.array(data2,order='C')
```


```python
arr2
```




    array([[1, 2, 3, 4],
           [5, 6, 7, 8]])




```python
arr2.ndim
```




    2




```python
arr2.shape
```




    (2, 4)




```python
arr1.dtype
```




    dtype('float64')




```python
arr2.dtype
```




    dtype('int32')




```python
#2、使用方法zeros(shape, dtype=float, order='C')生成全零数组。order参数表示是否以行优先（C样式）或列优先（Fortran样式）的顺序存储多维数据在内存中。
np.zeros((2,3),dtype='float32')
```




    array([[0., 0., 0.],
           [0., 0., 0.]], dtype=float32)




```python
#3、使用方法ones(shape, dtype=None, order='C')创建全1数组。
np.ones((3,3))
```




    array([[1., 1., 1.],
           [1., 1., 1.],
           [1., 1., 1.]])




```python
#3、使用方法empty(shape, dtype=float, order='C')创建没有初始化数值的数组。
np.empty((2,3,2))
```




    array([[[0., 0.],
            [0., 0.],
            [0., 0.]],
    
           [[0., 0.],
            [0., 0.],
            [0., 0.]]])




```python
#4、使用方法arange([start, ]stop, [step, ]dtype=None)创建序列化数组。
```


```python
np.arange(1,15,2)
```




    array([ 1,  3,  5,  7,  9, 11, 13])




```python
#5、使用函数numpy.asarray（a，dtype = None，order = None ），其中a可以是列表，元组列表，元组，元组元组，列表元组和ndarray。
```


```python
np.asarray((2,9)) #元组作为参数
```




    array([2, 9])




```python
np.asarray([1,2,3,4,5,6]) #列表作为参数
```




    array([1, 2, 3, 4, 5, 6])




```python
np.asarray(arr2) #ndarray作为参数
```




    array([[1, 2, 3, 4],
           [5, 6, 7, 8]])




```python
#6、使用函数numpy.ones_like（a，dtype = None，order ='K'，subok = True，shape = None ）其中参数a是一个ndarray,该方法返回一个与参数a相同形状的全1数组。
np.ones_like(arr2)
```




    array([[1, 1, 1, 1],
           [1, 1, 1, 1]])




```python
#7、同样也可以产生与给定数组相同形状的全0或未初始化的数组。zeros_like()或empty_like()
```


```python
#8、使用函数numpy.full(shape, fill_value, dtype=None, order='C')，产生一个给定形状给定给定填充之的数组。full_like()
np.full((2,3),[1,2,3],dtype=np.float64)
```




    array([[1., 2., 3.],
           [1., 2., 3.]])




```python
np.full((2,3),10)
```




    array([[10, 10, 10],
           [10, 10, 10]])



函数numpy.eye（N，M = None，k = 0，dtype = <class'float'>，order ='C' ）<br>
N：指定数组行数<br>
M:指定数组列数，如果省略，默认值为N<br>
K：指定数组的哪一条对角线为1，主对角线默认值为0，如果k大于零，表示以主对角线为基准，向上数，负数向下数。



```python
np.eye(4,4,k=1)
```




    array([[0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.],
           [0., 0., 0., 0.]])




```python
np.eye(4,4,k=-1)
```




    array([[0., 0., 0., 0.],
           [1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.]])



## 4.1.3 ndarray数据类型


```python
arr1.dtype
```




    dtype('float64')




```python
arr2.dtype
```




    dtype('int32')




```python
arr3=np.array([1,2,3,4,5])
```


```python
arr3.dtype
```




    dtype('int32')




```python
float_arr3=arr3.astype(np.float64)#使用arrayinstance.astype(dtype)强制转换数据类型
```


```python
float_arr3.dtype
```




    dtype('float64')




```python
arr4=np.array([3.7,-1.2,-2.6,0.5,12.9,10.1])
```


```python
arr4
```




    array([ 3.7, -1.2, -2.6,  0.5, 12.9, 10.1])




```python
arr4.astype(np.int32)
```




    array([ 3, -1, -2,  0, 12, 10])




```python
numeric_strings=np.array(['1.25','-9.6','42'],dtype=np.string_)
```


```python
numeric_strings.astype(np.float64)
```




    array([ 1.25, -9.6 , 42.  ])



## 4.1.4 NumPy数据运算

1、算数运算


```python
arr=np.array([[1.,2.,3.],[4.,5.,6.]])
```


```python
arr
```




    array([[1., 2., 3.],
           [4., 5., 6.]])




```python
arr*arr
```




    array([[ 1.,  4.,  9.],
           [16., 25., 36.]])




```python
arr-arr
```




    array([[0., 0., 0.],
           [0., 0., 0.]])




```python
1/arr
```




    array([[1.        , 0.5       , 0.33333333],
           [0.25      , 0.2       , 0.16666667]])




```python
arr**2
```




    array([[ 1.,  4.,  9.],
           [16., 25., 36.]])




```python
arr2=np.array([[0.,4.,1.],[7.,2.,12.]])
```

2、比较运算


```python
arr2
```




    array([[ 0.,  4.,  1.],
           [ 7.,  2., 12.]])




```python
arr2>arr
```




    array([[False,  True, False],
           [ True, False,  True]])




```python
arr2==arr
```




    array([[False, False, False],
           [False, False, False]])



## 4.1.5基础索引与切片

1、一维数组切片与索引


```python
arr=np.arange(10)
```


```python
arr
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
arr[5]
```




    5




```python
arr[5:8]
```




    array([5, 6, 7])




```python
arr[5:8]=12
```


```python
arr
```




    array([ 0,  1,  2,  3,  4, 12, 12, 12,  8,  9])




```python
arr_slice=arr[5:8]
```


```python
arr_slice
```




    array([12, 12, 12])




```python
arr_slice[1]=12345
```


```python
arr
```




    array([    0,     1,     2,     3,     4,    12, 12345,    12,     8,
               9])




```python
arr_slice[:]=64
```


```python
arr
```




    array([ 0,  1,  2,  3,  4, 64, 64, 64,  8,  9])



2、二维数组切片与索引


```python
arr2d=np.array([[1,2,3],[4,5,6],[7,8,9]])
```


```python
arr2d
```




    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])




```python
arr2d[2]
```




    array([7, 8, 9])




```python
arr2d[0][2]
```




    3




```python
arr2d[0,2]
```




    3



3、三位数组


```python
arr3d=np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
```


```python
arr3d
```




    array([[[ 1,  2,  3],
            [ 4,  5,  6]],
    
           [[ 7,  8,  9],
            [10, 11, 12]]])




```python
arr3d[0]
```




    array([[1, 2, 3],
           [4, 5, 6]])




```python
old_values=arr3d[0].copy()
```


```python
arr3d[0]=42
```


```python
arr3d
```




    array([[[42, 42, 42],
            [42, 42, 42]],
    
           [[ 7,  8,  9],
            [10, 11, 12]]])




```python
arr3d[0]=old_values
```


```python
arr3d
```




    array([[[ 1,  2,  3],
            [ 4,  5,  6]],
    
           [[ 7,  8,  9],
            [10, 11, 12]]])




```python
arr3d[1,0]
```




    array([7, 8, 9])




```python
x=arr3d[1]
```


```python
x
```




    array([[ 7,  8,  9],
           [10, 11, 12]])




```python
x[0]
```




    array([7, 8, 9])



4、数组的切片索引


```python
arr
```




    array([ 0,  1,  2,  3,  4, 64, 64, 64,  8,  9])




```python
arr[1:6]
```




    array([ 1,  2,  3,  4, 64])




```python
arr2d
```




    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])




```python
arr2d[:2]
```




    array([[1, 2, 3],
           [4, 5, 6]])




```python
arr2d[:2,1:]
```




    array([[2, 3],
           [5, 6]])




```python
arr2d[1,:2]
```




    array([4, 5])




```python
arr2d[:2,2]
```




    array([3, 6])




```python
arr2d[:,:1]
```




    array([[1],
           [4],
           [7]])




```python
arr2d[:2,1:]=0
```


```python
arr2d
```




    array([[1, 0, 0],
           [4, 0, 0],
           [7, 8, 9]])



## 4.1.6布尔索引


```python
names=np.array(['Bob','Joe','Will','Bob','Will','Joe','Joe'])
```


```python
data=np.random.randn(7,4)
```


```python
data
```




    array([[ 0.06146659, -0.47446356, -1.15488434, -0.29020102],
           [ 0.68336395, -0.8845612 , -0.13065605,  0.56647313],
           [-0.23412605,  0.01162633, -0.16014747, -2.03751934],
           [-1.0266747 , -0.80915552,  0.26744668,  1.17170668],
           [-0.60170946, -2.62083531, -1.65868751,  0.55332308],
           [-0.44011464, -0.96090314,  1.43606521, -0.79452421],
           [-1.14706693,  0.1110064 , -0.03264682, -0.75488254]])




```python
names
```




    array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'], dtype='<U4')




```python
names=='Bob'
```




    array([ True, False, False,  True, False, False, False])




```python
data[names=="Bob"]
```




    array([[ 0.06146659, -0.47446356, -1.15488434, -0.29020102],
           [-1.0266747 , -0.80915552,  0.26744668,  1.17170668]])




```python
data[names=='Bob',2:]
```




    array([[-1.15488434, -0.29020102],
           [ 0.26744668,  1.17170668]])




```python
data[names=='Bob',3]
```




    array([-0.29020102,  1.17170668])




```python
names!='Bob'
```




    array([False,  True,  True, False,  True,  True,  True])




```python
data[~(names=='Bob')]
```




    array([[ 0.68336395, -0.8845612 , -0.13065605,  0.56647313],
           [-0.23412605,  0.01162633, -0.16014747, -2.03751934],
           [-0.60170946, -2.62083531, -1.65868751,  0.55332308],
           [-0.44011464, -0.96090314,  1.43606521, -0.79452421],
           [-1.14706693,  0.1110064 , -0.03264682, -0.75488254]])




```python
cond=names=='Bob'
```


```python
data[~cond]
```




    array([[ 0.68336395, -0.8845612 , -0.13065605,  0.56647313],
           [-0.23412605,  0.01162633, -0.16014747, -2.03751934],
           [-0.60170946, -2.62083531, -1.65868751,  0.55332308],
           [-0.44011464, -0.96090314,  1.43606521, -0.79452421],
           [-1.14706693,  0.1110064 , -0.03264682, -0.75488254]])




```python
mask=(names=='Bob')|(names=='Will')
```


```python
mask
```




    array([ True, False,  True,  True,  True, False, False])




```python
data[mask]
```




    array([[ 0.06146659, -0.47446356, -1.15488434, -0.29020102],
           [-0.23412605,  0.01162633, -0.16014747, -2.03751934],
           [-1.0266747 , -0.80915552,  0.26744668,  1.17170668],
           [-0.60170946, -2.62083531, -1.65868751,  0.55332308]])




```python
data[data<0]=0
```


```python
data
```




    array([[0.06146659, 0.        , 0.        , 0.        ],
           [0.68336395, 0.        , 0.        , 0.56647313],
           [0.        , 0.01162633, 0.        , 0.        ],
           [0.        , 0.        , 0.26744668, 1.17170668],
           [0.        , 0.        , 0.        , 0.55332308],
           [0.        , 0.        , 1.43606521, 0.        ],
           [0.        , 0.1110064 , 0.        , 0.        ]])




```python
data[names!='Joe']=7
```


```python
data
```




    array([[7.        , 7.        , 7.        , 7.        ],
           [0.68336395, 0.        , 0.        , 0.56647313],
           [7.        , 7.        , 7.        , 7.        ],
           [7.        , 7.        , 7.        , 7.        ],
           [7.        , 7.        , 7.        , 7.        ],
           [0.        , 0.        , 1.43606521, 0.        ],
           [0.        , 0.1110064 , 0.        , 0.        ]])



## 4.1.7神奇索引


```python
arr=np.empty((8,4))
```


```python
for i in range(8):
    arr[i]=i
```


```python
arr
```




    array([[0., 0., 0., 0.],
           [1., 1., 1., 1.],
           [2., 2., 2., 2.],
           [3., 3., 3., 3.],
           [4., 4., 4., 4.],
           [5., 5., 5., 5.],
           [6., 6., 6., 6.],
           [7., 7., 7., 7.]])




```python
arr[[4,3,0,6]]
```




    array([[4., 4., 4., 4.],
           [3., 3., 3., 3.],
           [0., 0., 0., 0.],
           [6., 6., 6., 6.]])




```python
arr[[-3,-5,-7]]
```




    array([[5., 5., 5., 5.],
           [3., 3., 3., 3.],
           [1., 1., 1., 1.]])




```python
arr=np.arange(32).reshape((8,4))
```


```python
arr
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15],
           [16, 17, 18, 19],
           [20, 21, 22, 23],
           [24, 25, 26, 27],
           [28, 29, 30, 31]])




```python
arr[[1,5,7,2],[0,3,1,2]]
```




    array([ 4, 23, 29, 10])




```python
arr[[1,5,7,2]][:,[0,3,1,2]]
```




    array([[ 4,  7,  5,  6],
           [20, 23, 21, 22],
           [28, 31, 29, 30],
           [ 8, 11,  9, 10]])



## 4.1.8 数组转置和换轴


```python
arr=np.arange(15).reshape((3,5))
```


```python
arr
```




    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14]])




```python
arr.T
```




    array([[ 0,  5, 10],
           [ 1,  6, 11],
           [ 2,  7, 12],
           [ 3,  8, 13],
           [ 4,  9, 14]])




```python
arr=np.random.randn(6,3)
```


```python
arr
```




    array([[-1.19614599,  0.20477475, -0.96121814],
           [ 1.14695952, -0.58021723,  0.51202615],
           [-0.14661439, -0.49190247,  0.52312409],
           [ 0.77053985,  2.62471982, -0.7161493 ],
           [-1.28739847, -0.20169158,  0.64815335],
           [-1.27664208, -2.29204351,  1.65884128]])




```python
np.dot(arr.T,arr)
```




    array([[ 6.64871863,  4.36992163, -1.84366682],
           [ 4.36992163, 12.80384984, -6.56380036],
           [-1.84366682, -6.56380036,  5.14449688]])




```python
arr=np.arange(16).reshape((2,2,4))
```


```python
arr
```




    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7]],
    
           [[ 8,  9, 10, 11],
            [12, 13, 14, 15]]])




```python
arr.transpose((1,0,2))
```




    array([[[ 0,  1,  2,  3],
            [ 8,  9, 10, 11]],
    
           [[ 4,  5,  6,  7],
            [12, 13, 14, 15]]])




```python
arr
```




    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7]],
    
           [[ 8,  9, 10, 11],
            [12, 13, 14, 15]]])




```python
arr.swapaxes(1,2)
```




    array([[[ 0,  4],
            [ 1,  5],
            [ 2,  6],
            [ 3,  7]],
    
           [[ 8, 12],
            [ 9, 13],
            [10, 14],
            [11, 15]]])




```python

```


```python

```
