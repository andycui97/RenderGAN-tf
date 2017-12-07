RenderGAN Reimplementation in Tensorflow for CS294. 

`generate.py` contains the 3D model and `stepup.py` contains the main RenderGAN code. 

Make sure the following python dependencies are installed

```
h5py
multiprocess
pathos
Pillow
scipy
tensorflow
```

Before running make sure to download the Beestag h5py data and make sure to change `load_real_images` in `stepup.py` to load it correctly. 

Once that is done simply type
```
mkdir data
python generate.py
python stepup.py
```
