x=5
y="hello , world !"
print(x)
print(y)

x=-1
if x > 0:
    print("x is positive")
else:
    print("x is non-positive")

for i in range(1,5):
    print(i)

i=0
while i < 5:
    print(i)
    i+=1

def welcome(name):
    return "hello ,"+name
print(welcome("sai"))

class person:
    def __init__(self,name,age):
        self.name=name
        self.age=age
    def greet(self):
        return "hello,my name is " + self.name ,    "my age is "+ self.age
p1=person("sai","23")
print(p1.greet())


import math
print(math.sqrt(16))

from math import pi
print(pi)

try:
    print(10/1)
except ZeroDivisionError:
    print("cannot divide by zero")

with open('example.txt','w') as file:
    file.write("hello,world!")

with open('example.txt','r')as file:
    content=file.read()
    print(content)

mylist=[1,2,3,4,5]
print(mylist)
mylist[2]=7
print(mylist)

mytuple=(1,2,3,4,5)
print(mytuple)

mylist=[1,2,3,4,5]
mylist.append(6)
print(mylist)

mylist=[1,2,3]
mylist.append(4)
print(mylist)
mylist.remove(2)
print(mylist)

mytuple=(1,2,3,2)
print(mytuple.count(2))
print(mytuple.index(3))

mylist=[1,2,3]
mytuple=tuple(mylist)
print(mytuple)

mytuple=(1,2,3)
mylist=list(mytuple)
print(mylist)

"""import socket
s = socket.socket(socket.AF INET, socket.SOCK STREAM)
s.blind('localhost'12345)
s.listen(5)
packets=[]
def handleclient(clientsocket):
    while True:
        packet=clientsocket.rev(1024)
        if not packet:
            break
        packets.append(packet)
        if len(packets)>100:
            packets.pop(0)
    while True:
        clientsocket,addr=s.accept()
        handleclient(clientsocket)
        clientsocket.close()"""

class Dog:
    def __init__(self,name,age):
        self.name = name
        self.age = age

    def bark(self):
        return f"{self.name} says woof!"
mydog=Dog("buddy",5)
print(mydog.bark())

class animal:
    def __init__(self,name):
        self.name=name
    def speak(self):
        pass
class dog(animal):
    def speak(self):
        return f"{self.name} says woof!"
class cat(animal):
    def speak(self):
        return f"{self.name} says meow!"
Dog = dog("buddy")
Cat=cat("whiskers")
print(Dog.speak())
print(Cat.speak())

class person:
    def __init__(self,name,age):
        self.name=name
        self.age=age
    def getage(self):
        return self.age
    def setage(self,age):
        if age>0:
            self.age=age
Person=person("sai",23)
print(Person.name)
print(Person.getage())
Person.setage(22)
print(Person.getage())

class bird:
    def speak(self):
        return "some generic bird sound"
class parrot(bird):
    def speak(self):
        return "parrot says hello!"
class sparrow(bird):
    def speak(self):
        return "sparrow chirps!"
def makebirdspeak(bird):
    print(bird.speak())
Parrot=parrot()
Sparrow=sparrow()
makebirdspeak(Parrot)
makebirdspeak(Sparrow)

from abc import ABC, abstractmethod
class Shape(ABC):
    @abstractmethod
    def area(self):
        pass
class Rectangle(Shape):
     def __init__(self, width, height):
        self.width = width
        self.height = height
def area(self):
    return self.width * self.height
class Circle(Shape):
    def __init__(self,radius):
        self.radius=radius
    def area(self):
        return 3.14 * self.radius ** 2
    
rectangle = Rectangle(5,10)
circle=Circle(7)

print(rectangle.area())
print(circle.area())

import numpy as np
arr=np.array([1,2,3,4,5])
print(arr * 2)

import pandas as pd
data={'Name':['Alice','Bob','Charlie'],'Age':[25,30,35]}
df=pd.DataFrame(data)
print(df)

import matplotlib.pyplot as plt
plt.plot([1,2,3],[4,5,6])
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Simple Plot')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd
df=pd.DataFrame({'x':[1,2,3],'y':[4,5,6]})
sns.lineplot(data=df,x='x',y='y')
plt.show()

from scipy import stats
t_stat,p_value=stats.ttest_1samp([1,2,3,4,5],3)
print(f"T-statistic:{t_stat},P_value:{p_value}")

from sklearn.linear_model import LinearRegression
import numpy as np
model=LinearRegression()
x= np.array([[1],[2],[3],[4],[5]])
y= np.array([1,2,3,4,5])
model.fit(x,y)
print(model.predict([[6]]))


import numpy as np
import pandas as pd
import statsmodels.api as sm 
x=pd.DataFrame({'feature1':[1,2,3],'feature':[4,5,6]})
y=pd.Series([7,8,9])
x=sm.add_constant(x)
model=sm.OLS(y,x).fit()
print(model.summary())

import pandas as pd 
import plotly.express as px 
df=pd.DataFrame({'x':[1,2,3],'y':[4,5,6]})
fig=px.line(df,x='x',y='y',title='Interactive line plot')
fig.show()

import tensorflow as tf
a=tf.constant(2)
b=tf.constant(3)
c=a+b
print(c.numpy())

import torch as th
a=th.tensor(2)
b=th.tensor(3)
c=a+b
print(c)

import pandas as pd
s=pd.Series([1,3,5,7,9],index=['a','b','c','d','e'])
print(s)

import pandas as pd
data={'Name':['Alice','Bob','Charlie'],'Age':[25,30,35],'City':['New York','Paris','London']}
df=pd.DataFrame(data)
print(df)

import pandas as pd
data={'Name':['Alice','Bob','Charlie','Alice'],'Age':[25,30,35,25],'City':['New York','Paris','London','New York']}
df=pd.DataFrame(data)
print(df)
df['Age'].fillna(df['Age'].mean(),inplace=True) #fill missing values
df.drop_duplicates(inplace=True) #remove duplicates
df['Age']=df['Age'].astype(int) #convert data types
print(df)

import pandas as pd
data={'Product':['A','B','A','B','C'],
      'Category':['Electronics', 'Furniture', 'Electronics', 'Furniture', 'Kitchen'],
        'Sales': [100, 200, 150, 300, 250]}
df = pd.DataFrame(data)
print(df)
category_sales=df.groupby('Category')['Sales'].sum()
print(category_sales)


import pandas as pd
customers1 = pd.DataFrame({'CustomerID':[1,2,3], 
                         'Name':['Alice','Bob','Charlie'],
                         'City':['New York','Paris','London']})
customers2 = pd.DataFrame({'CustomerID':[2, 3, 4],
                           'Name': ['Bob', 'Charlie', 'David'],
                           'Country': ['France', 'UK', 'USA']})
merged_customers= pd.merge(customers1, customers2, on='CustomerID', how='inner')
print(merged_customers)

import pandas as pd
date_range = pd.date_range(start='1/1/2020', periods=5, freq='D')
stock_prices = pd.DataFrame({'Date': date_range, 'Price': [100, 110, 105, 115, 120]})
stock_prices.set_index('Date', inplace=True)
print(stock_prices)
stock_prices['Moving_Avg'] = stock_prices['Price'].rolling(window=3).mean()
print(stock_prices)

import matplotlib.pyplot as plt
import pandas as pd
data = {'Product': ['A', 'B', 'A', 'B', 'C'],
        'Category': ['Electronics', 'Furniture', 'Electronics', 'Furniture', 'Kitchen'],
        'Sales': [100, 200, 150, 300, 250]}
df = pd.DataFrame(data)
category_sales = df.groupby('Category')['Sales'].sum()
category_sales.plot(kind='bar')
plt.xlabel('Category')
plt.ylabel('Total Sales')
plt.title('Sales by Category')
plt.show()

import pandas as pd
data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
df = pd.DataFrame(data)
print(df)
df_normalized = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
print(df_normalized)

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
mnist = tf.keras.datasets.mnist 
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")
predictions = model.predict(test_images)
print(np.argmax(predictions[0]))
plt.imshow(test_images[0].reshape(28, 28), cmap=plt.cm.binary)
plt.title(f"Predicted: {np.argmax(predictions[0])}")
plt.show()

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  
        self.conv2 = nn.Conv2d(32, 64, 3, 1)  
        self.fc1 = nn.Linear(24*24*64, 128) 
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.flatten(x, 1)  
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(trainloader)}")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Test accuracy: {100 * correct / total}%")

images, labels = next(iter(testloader))
images, labels = images.to(device), labels.to(device)
output = model(images)
_, predicted = torch.max(output.data, 1)
print(f"Predicted: {predicted[0].item()}")
plt.imshow(images[0].cpu().numpy().squeeze(), cmap='gray')  
plt.title(f"Predicted: {predicted[0].item()}")
plt.show()
