import torch
import yaml
import cv2
from PIL import Image

from model import CodiClassifier
from loader import CodiTransform, CodiDataset


class CodiInference:
    def __init__(self, config_path):
        self.__set_configs(config_path)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.weight_path = self.configs['weight_path']
        self.transform = CodiTransform()
        self.get_results = {n: s for s, n in CodiDataset.string_dict.items()}

        self.__load_model()

    def __set_configs(self, config_path):
        with open(config_path) as f:
            self.configs = yaml.load(f, Loader=yaml.FullLoader)

    def __load_model(self):
        codi_classifier = CodiClassifier(self.configs)
        self.model = codi_classifier.get_model()
        self.model.load_state_dict(torch.load(self.weight_path))
        self.model.to(self.device)

    def inference(self, img_path):
        img_arr = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if img_arr.shape[2] == 4:
            # alpha channel만 발라내기
            alpha = img_arr[:, :, 3]
            # alpha channel의 값 중 0인 것들만 True로 발라내기
            mask = (alpha == 0)
            # 이미지의 alpha channel의 값이 0인 위치에 red, green, blue, alpha 값 모두 255로 넣어주기
            img_arr[:, :, :4][mask] = [255, 255, 255, 255]
            img_arr = cv2.resize(img_arr[:, :, :3], dsize=(img_arr.shape[1], img_arr.shape[0]),
                                 interpolation=cv2.INTER_LINEAR)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_arr)
        x_tensor = self.transform(img_pil).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            x_tensor.to(self.device)
            y_tensor = self.model(x_tensor)
            pred = y_tensor.argmax(dim=1, keepdim=True)
            result = self.get_results[pred.item()]

        return result


if __name__ == "__main__":
    import os

    codi_inference = CodiInference('./configs.yaml')

    files = list(filter(lambda x: x.split('.')[-1] in ['jpg', 'png'], os.listdir('../data/vetements/samples')))
    path_list = [os.path.join('../data/vetements/samples', f) for f in files]

    for path in path_list:
        print('path:', path)
        result = codi_inference.inference(path)
        print('result:', result)
        print('-' * 5)
