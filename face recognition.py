import cv2 as cv
import numpy as np
import os
from l1pca import l1_pca
import random
import matplotlib.pyplot as pp

def faceload(pathName,database):
    compiledImages =[]    
    for filename in os.listdir(pathName):
        #print(filename)
        if filename.endswith(".pgm"):            
            if(database == "ORL"):                
                currentImage  = cv.imread(pathName + filename, -1) #For grayscale only intensity is returned              
                resizeImage = cv.resize(currentImage,(50,50)) #Resizing images to 50x50 resolution
                flattenedImage = resizeImage.flatten() #vecotirizing training image           
                #Zero centering the data
                mean = np.mean(flattenedImage)
                flattenedImage = flattenedImage - mean
                compiledImages.append(flattenedImage)
            elif(database == "yale"):
                currentImage  = cv.imread(pathName + filename, -1) #For grayscale only intensity is returned              
                resizeImage = cv.resize(currentImage,(50,50)) #Resizing images to 50x50 resolution
                flattenedImage = resizeImage.flatten() #vecotirizing training image            
                mean = np.mean(flattenedImage) #intensity of the image                
                if(mean > 50): #filters out the darkest images
                    flattenedImage = flattenedImage - mean #Zero centering the data
                    compiledImages.append(flattenedImage)
    
    Xj = np.array(compiledImages)    
    np.random.shuffle(Xj)
    #print("Xj",Xj)
    #print("X",Xj.shape)
     
    if(database == "ORL"):        
        training_data = Xj[:7,:]    # 7 images for training
        testing_data = Xj[7:,:]     # 3 images for testing
    elif("yale"):
        training_data = Xj[:8,:]    # 8 images for training 
        testing_data = Xj[8:25,:]     # 17 images for testing
    
    compiled_testdata.append(testing_data)    
    return training_data.transpose()

def test_images(Qprop,X_test):
    total_tested = np.zeros(7,dtype=int)
    accuracy = np.zeros(7,dtype=int)
    for i in range(0,8):
        class_data = compiled_testdata[i]
        current_class = classes[i]
        for t in range(0,class_data.shape[0]):
            xt = np.expand_dims(class_data[t],axis=1)
            #run through nearest subspace algorithm 
            
            for rank in range(1,7):               
                class_found = nearest_subspace(Qprop,xt,rank)
                print("Actual class is ",current_class)
                print("Class matched to ",class_found)
                total_tested[rank] += 1
                if(current_class == class_found):
                    accuracy[rank] += 1
        
    return total_tested,accuracy                 

def nearest_subspace(Qprop,xt,rank_till):
    #norm_array = []
    flag = 1
    min_val = 0
    #min_rank = 0
    for j in range(0,8):
        current_class = classes[j]
        #for rank in range(1,7):
        Q_class = Qprop[j]            
        qj = Q_class[:rank_till]
        e1 = xt 
        e3 = np.matmul(qj.transpose(),qj)
        e2 = np.matmul(e3,e1)
        X = e1 - e2
        temp = np.linalg.norm(X, 2)
        if(flag == 1):
            min_val = temp
                #min_rank = rank
            match_class = current_class
            flag = 0
        
        if(temp < min_val):
            min_val = temp
                #min_rank = rank
            match_class = current_class
        
    #print("For class "+str(current_class)+" minimum found at class "+str(match_class)+"and rank "+str(min_rank))
        
    return match_class


# Running 50 experiments
collected_error = np.zeros(7,dtype=int)     
total_test = np.zeros(7,dtype=int)
database = "yale" #ORL for at&t db and yale for Ext Yale db

for test in range(1,10):    
    Qprop = []
    compiled_testdata = []
    Q_final = []
    
    # select 8 random classes from the database
    if(database == "ORL"):
        classes = random.sample(range(1,40), 8)
    else:
        a = np.array(range(1,39))
        a = np.delete(a,13) # DB missing class 14
        classes = np.random.choice(a,size=8,replace=True)
    
    # Calculating Qprop for all the class s1-40
    for j in classes:
        Qprop = []
        if(database == "ORL"):
            pathName = "D:/University Study/Artificial Intelligence/Face Recognition DB/att_faces/s"+str(j)+"/"
        elif(database == "yale"):
            pathName = "D:/University Study/Artificial Intelligence/Face Recognition DB/CroppedYale/yaleB"+str(j).zfill(2)+"/"
        
        print("Loading images from ",pathName)
        compiledImages = faceload(pathName,database)
        compiledQrop = []
        
        Qj = l1_pca(compiledImages, 10)
        if Qj is None:
            print("Qj is None")
            Qj = l1_pca(compiledImages, 10)        
        #Qj = np.expand_dims(Qj,axis=1)
        Qprop.append(Qj)
        
        for rank in range(2,7):
            temp_Qj = np.expand_dims(Qj,axis=1)
            temp = np.matmul(temp_Qj,temp_Qj.transpose())
            #print(temp.shape)
            temp2 =  np.matmul(temp,compiledImages)
            #print(temp2.shape)
            compiledImages = compiledImages - temp2
            #print("Modified data",compiledImages.shape)
            Qj = l1_pca(compiledImages, 10)
            if Qj is None:
                Qj = l1_pca(compiledImages, 10)
            Qprop.append(Qj)
            #print("Qprop rank "+str(rank),Qprop)
        
        t = np.array(Qprop)
        Q_final.append(t)
        
    
    Xj = compiledImages
    
    #print("Q_final",Q_final)    
    #print("Q_final shape",Q_final.shape)    
    
    compiled_testdata = np.array(compiled_testdata)
    #print("All testing data:",compiled_testdata.shape)
    #print("All testing data:",compiled_testdata)    
    total_tested,accuracy = test_images(Q_final,compiled_testdata)
    
    error = total_tested - accuracy
    collected_error = collected_error + error
    total_test = total_test + total_tested
    
collected_error = np.delete(error,0)
total_test = np.delete(total_test,0)
collected_error = collected_error*100
error_percent = collected_error/total_test


pp.ylabel('Error %') 
pp.xlabel('Rank of L1-PCA')
pp.ylim(0,100)
pp.plot([1,2,3,4,5,6],error_percent,'-x')
pp.show()