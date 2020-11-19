# ekstep-language-identification


mkdir -p /home/<user_name>/training_data

#Place training data inside /home/<user_name>/training_data
#looks like /home/<user_name>/training_data/train_hindi
#/home/<user_name>/training_data/train_english

#Build the image
docker build -t ekstep/ekstep-language-identification .

#Run the container
docker run --ipc=host --shm-size 16G --gpus all -it -v "/home/<user_name>/training_data:/training_data" ekstep/ekstep-language-identification:latest '/training_data/train_hindi' '/training_data/train_english' 
