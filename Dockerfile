

FROM python:latest


ARG csvFolder="Original Data | Fiji Organoid Measurements/"
ARG codeFolder="Code/"
ARG statFolder="Processed Data | Statistics/"
ARG target="./"
COPY ${csvFolder} ${csvFolder}
COPY ${codeFolder} ${codeFolder}
# COPY . . 

WORKDIR ./Code/

RUN pip install --upgrade pip
RUN pip install numpy

RUN apt-get update
#Dependencies to OpenCV
RUN apt-get install ffmpeg libsm6 libxext6  -y 
RUN pip install opencv-contrib-python

RUN pip install tifffile
RUN pip install scipy 
RUN pip install matplotlib


CMD python3 REV.py

# CMD ls -la 
