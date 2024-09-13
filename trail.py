import torch,torchvision,cv2
from PIL import Image
from collections import OrderedDict
from waternet.waternet.net import WaterNet
from waternet.waternet.data import transform
import numpy as np
def load_model(device:torch.device,
               ckpt_dir:str="/home/muahmmad/projects/Image_enhancement/waternet/weights/waternet_exported_state_dict-daa0ee.pt"):
    model=WaterNet()
    ckpt=torch.load(f=ckpt_dir,map_location=device,weights_only=True)
    print(ckpt.keys())
    model.load_state_dict(state_dict=ckpt)
    model=model.to(device=device)
    return model

def transform_array_to_image(arr):
    arr=np.clip(a=arr,a_min=0,a_max=1)
    arr=(arr*255.0).astype(np.uint8)
    return arr
def transform_image(img):
    trans=torchvision.transforms.Compose(transforms=[
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(size=(720,720),interpolation=torchvision.transforms.InterpolationMode.NEAREST_EXACT),
        torchvision.transforms.Normalize(mean=[0,0,0],
                                         std=[1,1,1])
    ])
    raw_image_tensor=trans(img)
    wb, gc, he=transform(img)
    wb_tensor=trans(wb)
    gc_tensor=trans(gc)
    he_tensor=trans(he)
    return {"X":torch.unsqueeze(input=raw_image_tensor,dim=0),
            "wb":torch.unsqueeze(input=wb_tensor,dim=0),
            "gc":torch.unsqueeze(input=gc_tensor,dim=0),
            "he":torch.unsqueeze(input=he_tensor,dim=0)}
def run():
    device = torch.device(device="cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    model.eval()
    image = cv2.imread(
        filename="/home/muahmmad/projects/Image_enhancement/Enhancement_Dataset/9898_no_fish_f000130.jpg")

    tensors = transform_image(img=image)
    raw_image_tensor = tensors["X"]
    wb_tensor = tensors["wb"]
    gc_tensor = tensors["gc"]
    he_tensor = tensors["he"]
    with torch.no_grad():
        raw_image_tensor = raw_image_tensor.to(device=device)
        wb_tensor = wb_tensor.to(device=device)
        gc_tensor = gc_tensor.to(device=device)
        he_tensor = he_tensor.to(device=device)
        print(he_tensor.device)
        pred = model(raw_image_tensor, wb_tensor, he_tensor, gc_tensor)
    pred = pred.squeeze_()
    pred = torch.permute(input=pred, dims=(1, 2, 0))
    output = pred.detach().cpu().numpy()
    output = transform_array_to_image(output)
    cv2.imshow(winname="org", mat=image)
    cv2.imshow(winname="pred", mat=output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""
run()
"""