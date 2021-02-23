import os
import torch
from torch import nn
from torch.autograd import Variable
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import utils
from arch import define_Gen, define_Dis
import numpy as np
from sklearn.metrics import mean_absolute_error
from skimage.metrics import peak_signal_noise_ratio
from calculate_fid import calculate_fid
import tensorflow as tf #to print shape of tensor with tf.shape() 

def test(args):

    transform = transforms.Compose(
        [transforms.Resize((args.crop_height,args.crop_width)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    dataset_dirs = utils.get_testdata_link(args.dataset_dir)

    a_test_data = dsets.ImageFolder(dataset_dirs['testA'], transform=transform)
    b_test_data = dsets.ImageFolder(dataset_dirs['testB'], transform=transform)


    a_test_loader = torch.utils.data.DataLoader(a_test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    b_test_loader = torch.utils.data.DataLoader(b_test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    Gab = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG='resnet_9blocks', norm=args.norm, 
                                                    use_dropout= not args.no_dropout, gpu_ids=args.gpu_ids)
    Gba = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG='resnet_9blocks', norm=args.norm, 
                                                    use_dropout= not args.no_dropout, gpu_ids=args.gpu_ids)

    utils.print_networks([Gab,Gba], ['Gab','Gba'])

    try:
        ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
        Gab.load_state_dict(ckpt['Gab'])
        Gba.load_state_dict(ckpt['Gba'])
    except:
        print(' [*] No checkpoint!')


#run test and calculate evaluation metrics   
    a_real_test = iter(a_test_loader)
    b_real_test=iter(b_test_loader)

    
    device = torch.device("cuda")
    batch_size= args.batch_size

    #real examples for plotting (save pic)
    a_real_example=(a_real_test.next()[0]).to(device)
    b_real_example=(b_real_test.next()[0]).to(device)

    Gab.eval()
    Gba.eval()
    
    list_T1_mae=[]
    list_T1_psnr=[]
    list_T1_fid=[]
    
 # for b test dataset - corresponds to T1 images
    for imagesT1, imagesT2 in zip(b_real_test, a_real_test):    
                 
          with torch.no_grad():
            
            imagesT1=imagesT1[0].to(device) #só o primeiro índice mas então o que têm os outros?
            imagesT2=imagesT2[0].to(device) 

            a_fake_test = Gab(imagesT1) #T2 translated
            b_recon_test = Gba(a_fake_test) #T1 reconstructed 
            
            b_fake_test = Gba(imagesT2) #T1 translated
            a_recon_test = Gab(b_fake_test) #T2 reconstructed
            
            
            imagesT1=imagesT1.cpu()
            #brec_to_cpu=b_recon_test.cpu() #T1 reconstructed 
            bfake_to_cpu=b_fake_test.cpu() #T1 translated 
            
            
            #brec_to_cpu = brec_to_cpu.view(batch_size, 3, 256, 256).numpy()
            bfake_to_cpu = bfake_to_cpu.view(batch_size, 3, 256, 256).numpy()

            
            imagesT1=np.squeeze(imagesT1) # squeezed to be [3, 256, 256] before was [1, 3, 256, 256]
            #brec_to_cpu=np.squeeze(brec_to_cpu) # squeezed to be [3, 256, 256] before was [1, 3, 256, 256]
            bfake_to_cpu=np.squeeze(bfake_to_cpu) # squeezed to be [3, 256, 256] before was [1, 3, 256, 256]
            
            
            imagesT1=imagesT1[1,:,:].numpy() #choose 1 channel of the RGB 
            #brec_to_cpu=brec_to_cpu[1,:,:] #choose 1 channel of the RGB 
            bfake_to_cpu=bfake_to_cpu[1,:,:] #choose 1 channel of the RGB 
            
            
            images_fid=imagesT1.reshape((1, 256, 256)) # check if it is this or reshape(1,256,256) - see AE_T1T2 the shape and size of the tensors before going in the MAE
            #brec_fid= brec_to_cpu.reshape((1, 256, 256))
            bfake_fid= bfake_to_cpu.reshape((1, 256, 256))
            
            #change this to calculate the MAE, PSNR and FID between b_real (from the dataset of T1 images real) and b_fake (the translated T1 images from the T2 slices)
            list_T1_mae.append(mean_absolute_error(imagesT1,bfake_to_cpu))
            list_T1_psnr.append(peak_signal_noise_ratio(imagesT1,bfake_to_cpu))
            list_T1_fid.append(calculate_fid(images_fid,bfake_fid))
                    
    
            
    # could add to see the shape/size of the list - should be flatten :
        
    #print mean of MAE, PSNR, FID       # compute the mean of the flatten array 
    print("Mean of MAE = " + str(np.mean(list_T1_mae)))
    print("Mean of PSNR = " + str(np.mean(list_T1_psnr)))
    print("Mean of FID = " + str(np.mean(list_T1_fid)))
    
    #print variance of MAE, PSNR, FID  # compute the variance of the flatten array 
    print("Variance of MAE = " + str(np.var(list_T1_mae)))
    print("Variance of PSNR = " + str(np.var(list_T1_psnr)))
    print("Variance of FID = " + str(np.var(list_T1_fid)))                     


    #Example for saving pic - just using the first image example of the datasets to plot the image 
    with torch.no_grad():
      #input is T2 images
      b_fake_example = Gba(a_real_example) # output is the translated T1 image from the inputed T2 slice
      a_recon_example = Gab(b_fake_example) # output is the reconstructed T2 slice 

      #input is T1 images
      a_fake_example = Gab(b_real_example) # output is the translated T2 image from the inputed T1 slice
      b_recon_example = Gba(a_fake_example) # output is the reconstructed T1 slice 


    # a_real_example= T2 real ; b_fake_example= T1 translated ; a_recon_example = T2 reconstructed | b_real_example= T1 real ; a_fake_example = T2 translated ; b_recon_example= T1 reconstructed
    pic = (torch.cat([a_real_example, b_fake_example, a_recon_example, b_real_example, a_fake_example, b_recon_example], dim=0).data + 1) / 2.0
    
    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)
    
    torchvision.utils.save_image(pic, args.results_dir+'/sample.jpg', nrow=3)
    
