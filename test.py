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

    
  # a_real_test = Variable(iter(a_test_loader).next()[0], requires_grad=True)
    # b_real_test = Variable(iter(b_test_loader).next()[0], requires_grad=True)
    # a_real_test, b_real_test = utils.cuda([a_real_test, b_real_test])
    
    a_real_test = iter(a_test_loader)
    b_real_test=iter(b_test_loader)
    #a_real_test, b_real_test = utils.cuda([a_real_test, b_real_test])
    #tensor=tf.shape(np.squeeze(b_real_test.next()[0]))
    #print(tensor)
    #print(tensor.numpy())

    
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
    for imagesT1 in b_real_test:    
                 
          with torch.no_grad():
            
            imagesT1=imagesT1[0].to(device)

            a_fake_test = Gab(imagesT1) 
            b_recon_test = Gba(a_fake_test) 
            
            
            #o que tinha no Google Colab - não sei se devo continuar com isto 
            imagesT1=imagesT1.cpu()
            #print('images T1 size b4 squeezed:', tf.shape(imagesT1))

            #br_to_cpu=b_real_test.cpu() #isto aqui não é b_real_test
            brec_to_cpu=b_recon_test.cpu()
            brec_to_cpu = brec_to_cpu.view(batch_size, 3, 256, 256)
            #print('tensor size:' , tf.shape(brec_to_cpu))
            brec_to_cpu = brec_to_cpu.numpy()
            #print('numpy size:' , brec_to_cpu.size)
            

            imagesT1=np.squeeze(imagesT1) # squeezed to be [3, 256, 256]
            #print('images T1 size squeezed:' , brec_to_cpu.size)
            brec_to_cpu=np.squeeze(brec_to_cpu) # squeezed to be [3, 256, 256]
            
            imagesT1=imagesT1[1,:,:].numpy() #choose 1 channel of the RGB 
            brec_to_cpu=brec_to_cpu[1,:,:] #choose 1 channel of the RGB 
            #print('output squeezed:' , brec_to_cpu.size)
            
            images_fid=imagesT1.reshape((1, 256, 256)) # check if it is this or reshape(1,256,256) - see AE_T1T2 the shape and size of the tensors before going in the MAE
            #print('images fid :' , images_fid.shape)

            brec_fid= brec_to_cpu.reshape((1, 256, 256))
            #print('outputs fid :' , brec_fid.shape)


          for batch in range(batch_size):
               
             list_T1_mae.append(mean_absolute_error(imagesT1[batch],brec_to_cpu[batch]))
             list_T1_psnr.append(peak_signal_noise_ratio(imagesT1[batch],brec_to_cpu[batch]))
             list_T1_fid.append(calculate_fid(images_fid[batch],brec_fid[batch]))
             
             
    
    # for a test dataset - corresponds to T2 images
    for imagesT2 in a_real_test:  
        
        with torch.no_grad():

            imagesT2=imagesT2[0].to(device)

            b_fake_test = Gba(imagesT2)
            a_recon_test = Gab(b_fake_test)  
            
            
    #print mean of MAE, PSNR, FID       
    print("Mean of MAE = " + str(np.mean(list_T1_mae)))
    print("Mean of PSNR = " + str(np.mean(list_T1_psnr)))
    print("Mean of FID = " + str(np.mean(list_T1_fid)))
    
    #print variance of MAE, PSNR, FID
    print("Variance of MAE = " + str(np.var(list_T1_mae)))
    print("Variance of PSNR = " + str(np.var(list_T1_psnr)))
    print("Variance of FID = " + str(np.var(list_T1_fid)))                     


    #Example for saving pic
    with torch.no_grad():
      #T2 images
      b_fake_example = Gba(a_real_example) 
      a_recon_example = Gab(b_fake_example)

      #T1 images
      a_fake_example = Gab(b_real_example) 
      b_recon_example = Gba(a_fake_example)


    # a_real_test= T2 real ; b_fake_test= T1 translated ; a_recon_test = T2 reconstructed | b_real_test= T1 real ; a_fake_test = T2 translated ; b_recon_test= T1 reconstructed
    pic = (torch.cat([a_real_example, b_fake_example, a_recon_example, b_real_example, a_fake_example, b_recon_example], dim=0).data + 1) / 2.0
    
    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)
    
    torchvision.utils.save_image(pic, args.results_dir+'/sample.jpg', nrow=3)
    
