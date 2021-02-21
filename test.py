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
import tf #to print shape of tensor with tf.shape() 

def test(args):

    transform = transforms.Compose(
        [transforms.Resize((args.crop_height,args.crop_width)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    dataset_dirs = utils.get_testdata_link(args.dataset_dir)

    a_test_data = dsets.ImageFolder(dataset_dirs['testA'], transform=transform)
    b_test_data = dsets.ImageFolder(dataset_dirs['testB'], transform=transform)


    a_test_loader = torch.utils.data.DataLoader(a_test_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    b_test_loader = torch.utils.data.DataLoader(b_test_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

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


    """ run """
    # a_real_test = Variable(iter(a_test_loader).next()[0], requires_grad=True)
    # b_real_test = Variable(iter(b_test_loader).next()[0], requires_grad=True)
    # a_real_test, b_real_test = utils.cuda([a_real_test, b_real_test])
    
    a_real_test = iter(a_test_loader)
    b_real_test=iter(b_test_loader)
    #a_real_test, b_real_test = utils.cuda([a_real_test, b_real_test])
    
    print(tf.shape(b_real_test.next()[0]))
    
    device = torch.device("cuda")
    
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
            #br_to_cpu=b_real_test.cpu() #isto aqui não é b_real_test
            brec_to_cpu=b_recon_test.cpu()
            
            # br_np=np.squeeze(br_to_cpu)
            # brec_np=np.squeeze(brec_to_cpu)
            
            # br_calc=br_np[1,:,:]
            # brec_calc=brec_np[1,:,:]
            
            
          for batch in range(batch_size=args.batch_size):
                
             list_T1_mae.append(mean_absolute_error(np.squeeze(imagesT1[batch]),np.squeeze(brec_to_cpu[batch])))
             list_T1_psnr.append(peak_signal_noise_ratio(np.squeeze(br_to_cpu[batch]),np.squeeze(brec_to_cpu[batch])))
             list_T1_fid.append(calculate_fid(np.squeeze(br_to_cpu[batch]),np.squeeze(brec_to_cpu[batch])))
             
             
    
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
            
    # a_real_test= T2 real ; b_fake_test= T1 translated ; a_recon_test = T2 reconstructed | b_real_test= T1 real ; a_fake_test = T2 translated ; b_recon_test= T1 reconstructed
    pic = (torch.cat([a_real_test[0], b_fake_test[0], a_recon_test[0], b_real_test[0], a_fake_test[0], b_recon_test[0]], dim=0).data + 1) / 2.0
    
    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)
    
    torchvision.utils.save_image(pic, args.results_dir+'/sample.jpg', nrow=3)
    
