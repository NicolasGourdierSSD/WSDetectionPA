RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app