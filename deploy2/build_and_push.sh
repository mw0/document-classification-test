#!/usr/bin/env bash

# The argument to this script is the image name. This will be used as the image on the local
# machine and combined with the account and region to form the repository name for ECR.
image=$1

if [ "$image" == "" ]
then
    echo "Usage: $0 <image-name>"
    exit 1
fi

chmod +x model/train
chmod +x model/serve

# Get the account number associated with the current IAM credentials
account=$(aws sts get-caller-identity --query Account --output text)
echo "account: $account"

if [ $? -ne 0 ]
then
    exit 255
fi


# Get the region defined in the current configuration (default to us-west-2 if none defined)
# us-west-1 == ireland
region=$(aws configure get region)
region=${region:-us-east-1}
echo "region: $region"

#fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"
fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}"
echo "fullname: $fullname"

# If the repository doesn't exist in ECR, create it.

aws ecr describe-repositories --repository-names "${image}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    echo "creating repository  ..."
    aws ecr create-repository --repository-name "${image}" > /dev/null
fi

# Get the login command from ECR and execute it directly
$(aws ecr get-login-password  --region ${region} --no-include-email)

# Build the docker image locally with the image name and then push it to ECR
# with the full name.

echo "building image ..."
docker build --no-cache -t ${image} .
docker tag ${image} ${fullname}

echo "pushing image ..."
docker push ${fullname}
echo "Done!"
