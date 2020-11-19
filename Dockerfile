FROM python:3.8.6


# install build utilities
RUN apt-get update && \
	apt-get install -y gcc make apt-transport-https ca-certificates build-essential
RUN apt-get install libsndfile1-dev
# check our python environment
RUN python3 --version
RUN pip3 --version

# set the working directory for containers
RUN mkdir /opt/ekstep-language-identification/
WORKDIR /opt/ekstep-language-identification/

# Installing python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy all the files from the project’s root to the working directory
COPY . /opt/ekstep-language-identification/
RUN ls -la /opt/ekstep-language-identification/*

## Running Python Application
#CMD ["python3", "/src/main.py"]
ENTRYPOINT ["/bin/bash", "invocation_script.sh"]
CMD ["hindi_path","english_path"]
