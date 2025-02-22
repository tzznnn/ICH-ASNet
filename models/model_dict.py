from models.segment_anything.build_sam import sam_model_registry

def get_model():
    model = sam_model_registry['vit_b'](checkpoint='./pretrained/sam_vit_b_01ec64.pth')
    return model
