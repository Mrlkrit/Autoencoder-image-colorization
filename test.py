import os
import torch
from network import Network
from read_data import CustomDataset
from save import saveImages
from torch.utils.data import DataLoader

if __name__ == "__main__":
    data_path = './data'
    save_path = './save_test'
    batch_size = 32
    num_workers = 6
    model_path = 'model_ep80.pt'

    if not os.path.exists(model_path):
        print("No such file.")
        exit()

    dataset = CustomDataset.getDatasetPath(data_path)
    print('Test dataset len: {}'.format(len(dataset['test'])))
    data_loader = {'test' : DataLoader(dataset['test'], batch_size = batch_size, shuffle = False, num_workers = num_workers)}

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Use GPU: {}'.format(str(device) != 'cpu'))

    network = Network().to(device)

    network.load_state_dict(torch.load(os.path.join(model_path),map_location = device))

    if not os.path.exists(save_path):
        os.makedirs(save_path)


    for idx, image in enumerate(data_loader['test']):

        image_L,image_Lab = image[:,0:1,:,:].float().to(device),image.float().to(device)

        generated_image_ab = network(image_L).detach()         
        generated_image_Lab = torch.cat([image_L, generated_image_ab], dim=1).to(device)
        print("index: {}".format(idx))
        saveImages(image_Lab,generated_image_Lab,os.path.join(save_path, 'test_{}.png'.format(idx)), show = False)