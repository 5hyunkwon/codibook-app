import os
import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset, random_split
from torchvision import transforms


class CodiDatasetLoader:
    def __init__(self, image_path):
        torch.manual_seed(0)

        file_list = [os.path.join(image_path, path, part) + '.png'
                     for path in os.listdir(image_path) if path != '.DS_Store'
                     for part in ['상의', '하의', '신발']]
        label_list = [part for path in os.listdir(image_path) if path != '.DS_Store'
                      for part in ['상의', '하의', '신발']]

        dataset = CodiDataset(file_list, label_list, CodiTransform())

        len_tot = len(file_list)
        len_train = int(0.8 * len_tot)
        len_val = len_tot - len_train

        train_ds, val_ds = random_split(dataset, [len_train, len_val])
        self.ds_dict = {'train': train_ds, 'val': val_ds}

    def __call__(self, data_type):
        return self.ds_dict[data_type]


class CodiTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def __call__(self, img):
        return self.transform(img)


class CodiDataset(Dataset):
    string_dict = {
        "상의": 0,
        "하의": 1,
        "신발": 2,
    }

    def __init__(self, x_list, y_list, transform):
        self.x_list = x_list
        self.y_list = list(map(self.string_to_vector, y_list))
        self.transform = transform
        self.length = len(self.x_list)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img_path = self.x_list[idx]
        img_arr = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img_arr.shape[2] == 4:
            # alpha channel만 발라내기
            alpha = img_arr[:, :, 3]
            # alpha channel의 값 중 0인 것들만 True로 발라내기
            mask = (alpha == 0)
            # 이미지의 alpha channel의 값이 0인 위치에 red, green, blue, alpha 값 모두 255로 넣어주기
            img_arr[:, :, :4][mask] = [255, 255, 255, 255]
            img_arr = cv2.resize(img_arr[:, :, :3], dsize=(img_arr.shape[1], img_arr.shape[0]), interpolation=cv2.INTER_LINEAR)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_arr)
        x_tensor = self.transform(img_pil)
        label = torch.LongTensor(self.y_list)[idx]
        return x_tensor, label

    @classmethod
    def string_to_vector(cls, value):
        return cls.string_dict.get(value, None)


if __name__ == "__main__":

    image_path = '/Users/ohhyunkwon/Documents/2021 study/projects/codibook/data/img'

    ds_loader = CodiDatasetLoader(image_path)
    train_ds = ds_loader('train')
    print(train_ds[0])
    print(len(train_ds))
    val_ds = ds_loader('val')
    print(val_ds[0])
    print(len(val_ds))

    from torch.utils.data import DataLoader
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
    print(len(train_dl))
    for x, y in train_dl:
        print(x.shape)
        print(type(x))
        print(y.shape)
        print(type(y))
        break
