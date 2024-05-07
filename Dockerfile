FROM python:3.10

# Set the working directory
WORKDIR /obj_tracking

# System updates and install
RUN apt-get update

# Install requirements
COPY ./requirements.txt /obj_tracking/requirements.txt
RUN pip install -r /obj_tracking/requirements.txt

# Copy the files
#COPY . .
CMD ["/bin/sh", "-c", "bash"]

#super-gradients
#filterpy
#opencv-python-headless
#jupyter