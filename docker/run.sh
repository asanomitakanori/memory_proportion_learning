#!/bin/bash
docker run \
    -d \
    --init \
    --rm \
    -it \
    --gpus=all \
    --ipc=host \
    --name=asanomi \
    --env-file=.env \
    --volume=$PWD:/workspace \
    --volume=/raid/asanomi/dataset:/dataset \
    asanomi_image:latest \
    fish