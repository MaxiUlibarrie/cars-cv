import os 
import shutil

NEW_TRAINED_MODELS = "new_trained_models/"
MODELS = "models/"
MODELS_DEPLOY = "models_deploy/"

new_models = [ v for v in os.listdir(NEW_TRAINED_MODELS) if v.startswith("V") ] 

for v in new_models:
    try:
        os.mkdir(MODELS_DEPLOY + v)
    except:
        pass
    shutil.copy(NEW_TRAINED_MODELS + v + "/weights/best.pt", MODELS_DEPLOY + v + "/best.pt")
    os.system(f"cp -a {NEW_TRAINED_MODELS + v} {MODELS}")
    os.system(f"rm -rf {NEW_TRAINED_MODELS + v}/")
