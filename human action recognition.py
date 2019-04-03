import cv2
import os
import numpy as np
from l1pca import l1_pca
import matplotlib.pyplot as pp

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def test_images(Qprop,X_test):
    total_tested = np.zeros(7,dtype=int)
    accuracy = np.zeros(7,dtype=int)
    for i in range(0,len(classes)):
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
    for j in range(0,len(classes)):
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
    min_distance[np.array(classes) == match_class] = min_val
    return match_class

#Testing for Robustness
def test_robustness():
    robustClasses = []
    path = "D:/University Study/Artificial Intelligence/Human Action DB/Weizzman/robust_deform/"
    videoFiles = os.listdir(path)
    X = []
    for files in videoFiles:        
        # Create a VideoCapture object and read from input file
        # If the input is the camera, pass 0 instead of the video file name
        videoPath = path  + "/" + files
        robustClasses.append(files)
        print(videoPath)
        cap = cv2.VideoCapture(videoPath)
        # Check if camera opened successfully
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")        
        
        c = -1
        timestamp = 1
        ret, frame = cap.read()
        frame = image_resize(frame, width = 60) #Resizing images
            
        prev_frame = frame.copy()
        h, w = frame.shape[:2]
        motion_history = np.zeros((h, w), np.float32)
        # Read until video is completed
        while(cap.isOpened()):            
            c += 1
            ret, frame = cap.read()          
            if c%4 == 0:                
                if ret == True:                    
                    frame = image_resize(frame, width = 60) #Resizing images
                    frame_diff = cv2.absdiff(frame, prev_frame)
                    gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
                    #thrs = cv2.getTrackbarPos('threshold', 'motempl')
                    ret, motion_mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
                    cv2.motempl.updateMotionHistory(motion_mask, motion_history, timestamp, MHI_DURATION)   
                    #fgmask = fgbg2.apply(frame)
                    print("frame",motion_mask.shape)
                    #motion_array.append(fgmask)
                    # Display the resulting frame
                    cv2.imshow('fgmask',motion_mask)
                    # Press Q on keyboard to  exit
                    if cv2.waitKey(25) & 0xFF == ord('q'):                    
                        break             
                     # Break the loop
                else:                                         
                    break    
            # When everything done, release the video capture object
        cap.release() 
        motion_history[motion_history == 1] = 255
        cv2.imwrite('RobustMotion_History'+files+'.jpg',motion_history)
            
        #motion_history flatten for each action
        flattened_MI = motion_history.flatten()
            
        #Keep on appending
        X.append(flattened_MI)
            
        # Closes all the frames
        cv2.destroyAllWindows()
        #Split into training and testing
    print(robustClasses)    
     
    Xj = np.array(X)  
    print("Shape X",Xj.shape)
    #np.random.shuffle(Xj)           
    robustImages = Xj
    print("Robust Images Cluster:",robustImages.shape)       

    Qprop = np.array(Q_final)
    print("Robust Qprop",Qprop.shape)
    
    total_tested = np.zeros(7,dtype=int)
    accuracy = np.zeros(7,dtype=int)
    for i in range(0,len(robustClasses)):        
        #class_data = robustImages
        robust_class = robustClasses[i]       
        xt = np.expand_dims(robustImages[i],axis=1)
            #run through nearest subspace algorithm            
        for rank in range(1,7):            
            flag = 1
            min_val = 0
            
            #Finds nearest subspace for rank k
            for j in range(0,len(classes)):
                current_class = classes[j]
                #for rank in range(1,7):
                Q_class = Qprop[j]          
                qj = Q_class[:rank]
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
                    match_class = current_class
            
            #matched class founded
            print("For class "+str(robust_class)+" minimum found at class "+str(match_class) + " at rank "+str(rank))
            print("Min j^hat Value =",min_val)
            total_tested[rank] += 1
            if(match_class == 'walk'):
                accuracy[rank] += 1
    
    print("total",total_tested)
    print("accuracy",accuracy)
    error = total_tested - accuracy
    collected_error = np.delete(error,0)
    e2 = np.delete(total_tested,0)
    e1 = collected_error*100
    percent_error = e1/e2
        
    print("Error in Robustness",percent_error)

MHI_DURATION = 30
DEFAULT_THRESHOLD = 32
MAX_TIME_DELTA = 0.25
MIN_TIME_DELTA = 0.05

collected_error = np.zeros(7,dtype=int)
total_test = np.zeros(7,dtype=int)
min_distance = np.zeros(3,dtype=int)

path = "D:/University Study/Artificial Intelligence/Human Action DB/Weizzman/"

#Action selected
classes = ["walk","jack","bend"]
#classes = ["walk"]
for experiment in range(1,2):
    print("Starting experiment %d............",experiment)
    combines_test = []
    compiled_testdata = []
    Q_final = []
    
    for action in classes:
        Qprop = []
        #File list
        videoFiles = os.listdir(path + action)
        X = [] 
        for files in videoFiles:
            # Create a VideoCapture object and read from input file
            # If the input is the camera, pass 0 instead of the video file name
            videoPath = path + action + "/" + files
            print(videoPath)
            cap = cv2.VideoCapture(videoPath)
            # Check if camera opened successfully
            if (cap.isOpened()== False): 
              print("Error opening video stream or file")
            
            motion_array = []
            c = -1
            timestamp = 1
            ret, frame = cap.read()
            frame = image_resize(frame, width = 60) #Resizing images
            
            prev_frame = frame.copy()
            h, w = frame.shape[:2]
            motion_history = np.zeros((h, w), np.float32)
            # Read until video is completed
            while(cap.isOpened()):
              # Capture frame-by-frame
              c += 1
              ret, frame = cap.read()          
              if c%4 == 0:                  
                  if ret == True:
                    frame = image_resize(frame, width = 60) #Resizing images
                    frame_diff = cv2.absdiff(frame, prev_frame)
                    gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
                    #thrs = cv2.getTrackbarPos('threshold', 'motempl')
                    ret, motion_mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
                    cv2.motempl.updateMotionHistory(motion_mask, motion_history, timestamp, MHI_DURATION)   
                    #fgmask = fgbg2.apply(frame)
                    print("frame",motion_mask.shape)
                    #motion_array.append(fgmask)
                    # Display the resulting frame
                    cv2.imshow('fgmask',motion_mask)
                    # Press Q on keyboard to  exit
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                      break             
                  # Break the loop
                  else: 
                    break    
            # When everything done, release the video capture object
            cap.release() 
            motion_history[motion_history == 1] = 255
            cv2.imwrite('Motion_History'+files+'.jpg',motion_history)
            
            #motion_history flatten for each action
            flattened_MI = motion_history.flatten()
            
            #Keep on appending
            X.append(flattened_MI)
            
        # Closes all the frames
        cv2.destroyAllWindows()
        #Split into training and testing
        Xj = np.array(X)  
        print("Shape X",Xj.shape)
        np.random.shuffle(Xj)
        training_data = Xj[:7,:] #7 for training
        testing_data = Xj[7:,:]  #2 for testing
        compiled_testdata.append(testing_data)    
        compiledImages = np.transpose(training_data)
        #Run it through L1-PCA    
        Qj = l1_pca(compiledImages, 10) 
        if Qj is None:
            Qj = l1_pca(compiledImages, 10)
        print(Qj.shape)
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
                
        t = np.array(Qprop)
        Q_final.append(t)       #Append Qprops to Qfinal
        
        
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
    
    
test_robustness()


