import os
import torch
from torch.utils.data import DataLoader
from network import Network
from read_data import CustomDataset
from save import saveImages


"""MAIN"""
if __name__ == "__main__":

    path = './data'
    batch_size = 32
    num_workers = 6
    learning_rate = 0.001
    save_path = './save'
    start_epoch = 0
    end_epoch = 200
    save_frequency = 4;

    datasets = CustomDataset.getDatasetPath('./data')

    for phase in ['train','val','test']:
        print('{} dataset len: {}'.format(phase, len(datasets[phase])))

    data_loader = {
        'train': DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': DataLoader(datasets['val'], batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'test': DataLoader(datasets['test'], batch_size=batch_size, shuffle=False, num_workers=num_workers)
        }



    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Use GPU: {}'.format(str(device) != 'cpu'))

    network = Network().to(device)

    optimizer = torch.optim.Adam(network.parameters(), lr = learning_rate)
    loss_fn = torch.nn.MSELoss()

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if start_epoch>0:
        network.load_state_dict(torch.load(os.path.join(save_path,'model_ep{}.pt'.format(start_epoch-1)),map_location = device))


    for epoch in range(start_epoch,end_epoch):
        print('\n========== EPOCH {} =========='.format(epoch))

        for phase in ['train' , 'val']:

            if phase == 'train':
                print('TRAINING:')
            else:
                print('VALIDATION:')

            for idx, image in enumerate(data_loader[phase]):
                
                image_L,image_Lab = image[:,0:1,:,:].float().to(device),image.float().to(device)

                if phase == 'train':optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    generated_image_ab = network(image_L)
                    
                    generated_image_Lab = torch.cat([image_L, generated_image_ab], dim=1).to(device)

                    loss = loss_fn(image_Lab[:,1:,:,:],generated_image_ab)

                    if phase == 'train': 
                        loss.backward()
                        optimizer.step()

                if phase == 'val' and idx == 0:
                    sample_real_img_lab = image_Lab
                    sample_fake_img_lab = generated_image_Lab

            print('Network loss = {:.4f}'.format(loss))

            if phase == 'val':
                if epoch%save_frequency == 0 or epoch == end_epoch-1:
                    model_save_path = os.path.join(save_path, 'model_ep{}.pt'.format(epoch))
                    torch.save(network.state_dict(),model_save_path)
                    print("Model saved.")

                    saveImages(sample_real_img_lab, sample_fake_img_lab, os.path.join(save_path, 'images_ep{}.png'.format(epoch)),show =False)
           