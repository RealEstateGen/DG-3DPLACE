I am making a simple web app with frontend

must maintain a simple sqlite database

project is RealEstateGen DG-3GDS

diffusion guided 3dgs

This is the workflow. 


1. user must be able to upload a video containing the scene. this is a mp4 file. then after uploading the video it must be added to db and the video must be saved
Then user must be able to click on the video and preview and create a 3dgs scene. scene outputs must be stored. remember use polling until the 3dgs process is completed from the api.

Then user must be able to click on a scene and preview it intractively, we must be able to capture a snapshop (frame) of selected scene from a button intractively when use changes the camera position.


sudo docker run --gpus all -v $(pwd)/room:/workspace \
  nerfstudio/nerfstudio:latest \
  ns-train splatfacto --data /workspace/data --output-dir /workspace/output/scene \
  --max-num-iterations 7000 colmap


First run some tests with sample videos

videos
scenes
captured_images

captured images must show at the bottom of the intractive area. when clicking the captured scene user must be able to open it 

frontend must use react and backend fastapi

Use these commands

sudo docker run --gpus all \
  -v $(pwd)/room:/workspace \
  nerfstudio/nerfstudio:latest \
  ns-train splatfacto \
  --data /workspace/data \
  --output-dir /workspace/output/my_scene \
  --max-num-iterations 7000 \
  colmap \
  --colmap-path sparse/0 \
  --images-path images \
  --downscale-factor 1
```

1. Create project directory
mkdir -p room/new_scene/data/images
cd room/new_scene

2. Extract frames from video
sudo docker run --rm \
  -v $(pwd):/workspace \
  nerfstudio/nerfstudio:latest \
  ns-process-data video \
  --data /workspace/input_video.mp4 \
  --output-dir /workspace/data

3. Train on the processed data
sudo docker run --gpus all \
  -v $(pwd):/workspace \
  nerfstudio/nerfstudio:latest \
  ns-train splatfacto \
  --data /workspace/data \
  --output-dir /workspace/output \
  --max-num-iterations 7000 \
  colmap
```


First give me the plan. run some tests on yourself on sample videos like creating colmap and splat and others them if all good clean them by yourself