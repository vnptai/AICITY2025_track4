from mmdet.apis import DetInferencer
import warnings
warnings.filterwarnings('ignore')

model = '../projects/CO-DETR/configs/codino/test.py'
checkpoint = '../weights/epoch_13.pth'
image = '/media/hungdv/Source/Data/ai-city-challenge-2024/track4/fisheye_final/other_dataset/images/11_jpg.rf.05030e2971cfffdd8ac99b85bfe79222.jpg'
device = 'cuda:0'
inferencer = DetInferencer(model, checkpoint, device)
inferencer(image,
pred_score_thr=.3,
out_dir='../temp/',
show=False,
print_result=False)