
""" Dataset loader for Image-to-Image translation using T1 and T2 images from dHCP dataset - supervised learning"""

import os
import torch 
import os
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image


#For image-to-image translation T2 to T1 images (because we have less T1 images than T2) 
#Images are T2 slices and labels are T1 slices 
class MedicalData(torch.utils.data.Dataset):
    
    # initialise the class based on the folder containing the data and the project dataframe
    def __init__(self, folder_T1='',folder_T2='', transform=None):
    
    #provavelemente mudar isto para que os pacientes estejam alinhados uns com os outros - ver folder Emma
        self.folder_T1= folder_T1
        self.folder_T2 = folder_T2
        self.images_list = os.listdir(self.folder_T2)
        self.labels_list = os.listdir(self.folder_T1)
        self.transform=transform
        
        return
        
    
    def __len__(self):
        
        # return the number of examples in the folder
        return len(self.images_list)
    
        
    def __getitem__(self,idx):
       
        #load images
        os.chdir(self.folder_T2)
        image=np.load(self.images_list[idx],allow_pickle=True) #slice
        os.chdir(self.folder_T1)
        label=np.load(self.labels_list[idx],allow_pickle=True) #slice
      
      
        #Convert to tensors and reshape to expected dimensions - Image 256x256 
        im=np.pad(image,30)
        im=im[:,16:272]
        
        lab=np.pad(label,30)
        lab=lab[:,16:272]
        
        img_tensor = torch.from_numpy(im).to(torch.float) 
        lab_tensor= torch.from_numpy(lab).to(torch.float)
        
        #has to be in shape C x H x W - 1x256x256
        img_tensor=img_tensor.unsqueeze(0)
        lab_tensor=lab_tensor.unsqueeze(0) 
        
        
        # convert to JPG and return
        sample = img_tensor
        label = lab_tensor
        
        if self.transform:
            sample=self.transform(sample)
            label=self.transform(label)
            
        return sample , label 
    
""" Uncomment next section to test if Class Medical Data is working """
    
#set directory for model
# os.chdir('C:/Users/helen/OneDrive/Área de Trabalho/I2Itranslation/Image-translation')

# folder_slicesT1='C:/Users/helen/OneDrive/Área de Trabalho/I2Itranslation/Image-translation/Final slices T1'
# folder_slicesT2= 'C:/Users/helen/OneDrive/Área de Trabalho/I2Itranslation/Image-translation/Final slices T2'

# #create MedicalData model 
# model_dataset= MedicalData(folder_T1=folder_slicesT1, folder_T2=folder_slicesT2)    

# #check if our model is correct 
# lenght_data=model_dataset.__len__()

# get_sample=model_dataset.__getitem__(1)
    
# aa=transforms.ToPILImage()(get_sample[0])
# aa1=transforms.ToPILImage()(get_sample[1])

# b=transforms.Grayscale()(aa)
# b1=transforms.Grayscale()(aa1)

# plt.imshow(b, cmap=plt.cm.gray)
# plt.imshow(b1, cmap=plt.cm.gray)



'''Save images as png for CycleGAN (same number of T1 and T2 slices - paired '''

# save images in folder from the Medical Dataset as png images 

#folder for slices in numoy arrays 
folder_slicesT1='C:/Users/helen/OneDrive/Área de Trabalho/I2Itranslation/Image-translation/Final slices T1'
folder_slicesT2= 'C:/Users/helen/OneDrive/Área de Trabalho/I2Itranslation/Image-translation/Final slices T2'

#create MedicalData model 
model_dataset= MedicalData(folder_T1=folder_slicesT1, folder_T2=folder_slicesT2) 

#check lenght dataset (must be 2790)
lenght_data=model_dataset.__len__()

#list with filenames 
list_names = os.listdir(folder_slicesT1)  #both T1 and T2 folders have the same names for the slices so it can be either folder_slicesT1 or T2


for i in range(lenght_data):  
    
    get_sampleT2=model_dataset.__getitem__(i)[0] #get all T2 images
    get_sampleT1=model_dataset.__getitem__(i)[1] #get all T2 images
    
    list_names[i]= list_names[i].strip('.npy')
    name_slices=list_names[i] 

    toimageT2=transforms.ToPILImage()(get_sampleT2)
    toimageT1=transforms.ToPILImage()(get_sampleT1)
    
    os.chdir('C:/Users/helen/OneDrive/Área de Trabalho/I2Itranslation/CycleGAN/T2 slices png')
    png_T2= toimageT2.save(str(name_slices) + '.png') #save in folder as jpg image 

    os.chdir('C:/Users/helen/OneDrive/Área de Trabalho/I2Itranslation/CycleGAN/T1 slices png')
    png_T1= toimageT1.save(str(name_slices) + '.png') #save in folder as jpg image 
    
