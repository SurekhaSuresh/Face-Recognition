# In[1]:


from PIL import Image
from pathlib import Path
import numpy as np
import matplotlib.image as mpimage
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# • Start with 𝑀 training images, each of size 𝑁 × 𝑁 pixels.

tr = "Face dataset/Training/"
ts = "Face dataset/Testing/"
training_images = [tr+"subject01.happy.jpg",tr+"subject02.normal.jpg",tr+"subject03.normal.jpg",tr+"subject07.centerlight.jpg",tr+"subject10.normal.jpg",tr+"subject11.normal.jpg",tr+"subject14.normal.jpg",tr+"subject15.normal.jpg"]
training_images_names = ["Subject1","Subject2","Subject3","Subject7","Subject10","Subject11","Subject14","Subject15"]
test_images = [ts+"subject01.normal.jpg",ts+"subject07.happy.jpg",ts+"subject07.normal.jpg",ts+"subject11.happy.jpg",ts+"subject14.happy.jpg",ts+"subject14.sad.jpg"]
testing_images_names = ["Subject1","Subject7","Subject7","Subject11","Subject14","Subject14"]
testing_images_count = len(test_images)


# In[3]:


# • For each training image i, the rows are stacked together to form a column vector 𝑅𝑖 of dimension 𝑁2.

tr_image_vectors = []
for index,training_image in enumerate(training_images):
    tr_image_vector = mpimage.imread(training_image)
    n1, n2 = tr_image_vector.shape
    tr_image_vector = tr_image_vector.reshape(n1 * n2, )
    tr_image_vectors.append(tr_image_vector)
tr_image_vectors = np.asarray(tr_image_vectors)


# In[4]:


# • The mean face m is computed by taking the average of the M training face images.

meanMatrix = tr_image_vectors.mean(axis=0)
meanFace = meanMatrix.reshape(n1, n2)
plt.imshow(meanFace, cmap='gray')
plt.savefig("meanFace.jpg")
plt.show()


# In[5]:


# • We then subtract the mean face m from each training face 𝑅𝑖 = 𝑅𝑖 − 𝑚

tr_image_vectors = np.subtract(tr_image_vectors, meanMatrix)


# In[6]:


# • Put all training faces into a single matrix A (dimension = 𝑁2 × 𝑀): [𝑅1 𝑅2 ........𝑅𝑀]

Amatrix = tr_image_vectors.transpose()


# In[7]:


# • Compute the transpose of the matrix A as Amatrix_T

Amatrix_T = Amatrix.transpose()


# In[8]:


# • We can find up to M largest eigenvalues by this method – Find eigenvalues of 𝐿 = 𝐴𝑇A. 𝐿 is of dimension 𝑀 × 𝑀.

Lmatrix = np.dot(Amatrix_T,Amatrix)


# In[9]:


# • Put eigenvectors of 𝐿 into a single matrix 𝑉. ev contains the eigen values.

ev, Vmatrix = np.linalg.eig(Lmatrix)


# In[10]:


# • The 𝑀 largest eigenvectors of 𝐶 can be found by 𝑈 = 𝐴𝑉; 𝑈 then contains 𝑀 eigenfaces and has dimension 𝑁2 × 𝑀.

Umatrix = np.dot(Amatrix,Vmatrix)


# In[11]:


# • Compute the transpose of the matrix U as Umatrix_T

Umatrix_T = Umatrix.transpose()
for index,element in enumerate(tr_image_vectors):
    eigenFace = Umatrix_T[index].reshape(n1, n2)
    plt.imshow(eigenFace, cmap='gray')
    plt.savefig("eigenFace_" + str(index+1) + ".jpg")
    plt.show()


# In[12]:


# • Each training face can then be projected onto the face space to obtain its eigenface coefficients Ω𝑖=𝑈𝑇𝑅𝑖 for𝑖=1to𝑀.

coeffs = []
for index,tr_image_vector in enumerate(tr_image_vectors):
    coeff = np.dot(Umatrix_T,tr_image_vector)
    coeffs.insert(index, coeff)
    print("\nEigen Face coefficients of the training image ", index+1 , "\n" )
    print(coeff)


# In[13]:


# • Subtract mean face m from input face I: 𝐼=𝐼−𝑚.

ts_image_vectors = []
for index,test_image in enumerate(test_images):
    ts_image_vector = mpimage.imread(test_image)
    n3, n4 = ts_image_vector.shape
    ts_image_vector = ts_image_vector.reshape(n3 * n4, )
    ts_image_vectors.append(ts_image_vector)
ts_image_vectors = np.asarray(ts_image_vectors)

ts_image_vectors = np.subtract(ts_image_vectors, meanMatrix)


# In[14]:


correctRecognitions = 0
for index,ts_image_vector in enumerate(ts_image_vectors):
    
    # • Compute its projection onto face space to obtain eigenface coefficients Ω𝐼 = 𝑈𝑇𝐼.
    coeffI = np.dot(Umatrix_T,ts_image_vector)
    print("Eigen face coefficient for the input test image - " + testing_images_names[index] + " :\n")
    print(coeffI)
    
    smallDist = np.inf
    resultIndex = -1
    
    for i,coeff in enumerate(coeffs):
        
        # • Compute distance between input face and training images in the face space. 
        # (distance between eigenface coefficients.) 
        # 𝑑𝑖 = dist(Ω𝐼,Ω𝑖), for 𝑖 = 1,2,...,𝑀.
        dist = np.linalg.norm(coeffI-coeff)
        
        if dist < smallDist:
            smallDist = dist
            resultIndex = i
            
    tested_image = testing_images_names[index]
    testimg = Image.open(test_images[index])
    print("Input Test Image : " + tested_image + "\n")
    plt.imshow(testimg, cmap='gray')
    plt.show()
    
    if(resultIndex != -1):
        matched_image = training_images_names[resultIndex]
        print("Matched Image : " + matched_image)
        trimg = Image.open(training_images[resultIndex])
        plt.imshow(trimg, cmap='gray')
        plt.show()
        if(tested_image == matched_image):
            correctRecognitions += 1
    else:
        print("No matched image found.")


# In[15]:


# • Compute the accuracy of the face recognition algorithm using the formula : (# correct recognitions / # test images) * 100.

accuracy = (correctRecognitions / testing_images_count) * 100
print("Accuracy is " , accuracy , "%")