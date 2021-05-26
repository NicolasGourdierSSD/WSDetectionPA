apt-get update
apt install libopencv-dev python3-opencv
apt-get install libgl1-mesa-glx
apt-get install ffmpeg libsm6 libxext6
apt-get install libglib2.0-0
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app