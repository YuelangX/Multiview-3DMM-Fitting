from lib.face_models.BFMModule import BFMModule
from lib.face_models.FaceVerseModule import FaceVerseModule
from lib.face_models.FLAMEModule import FLAMEModule

def get_face_model(face_model, batch_size, device):
    if face_model == 'BFM':
        return BFMModule(batch_size).to(device)
    elif face_model == 'FaceVerse':
        return FaceVerseModule(batch_size).to(device)
    elif face_model == 'FLAME':
        return FLAMEModule(batch_size).to(device)
    else:
        raise "face_model should be one of {'BFM', 'FaceVerse', 'FLAME'}"