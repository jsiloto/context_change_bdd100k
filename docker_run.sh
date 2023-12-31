#!/bin/bash

############ Input Boilerplate ############
usage() { echo "Usage: $0 [-g <gpu>]" 1>&2; exit 1; }

while getopts ":g:" o; do
    case "${o}" in
        g)
            g=${OPTARG}
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))

if  [ -z "${g}" ]; then
    usage
fi

if  [ -z "${DATASET}" ]; then
    echo "DATASET variable must be defined"
fi

if  [ -z "${YOLO}" ]; then
    echo "YOLO variable must be defined"
fi


################################################
echo "GPUS ${g}"

COMMAND="docker run --rm -it \
            --shm-size=32G  \
            -v $DATASET:/data/datasets/  \
            -v $YOLO:/work/yolov5  \
            -v $PWD:/work/context_change_bdd100k -w /work \
            -u $(id -u):$(id -g)  \
            --network=host \
	    --userns=host  \
	    --gpus='"device=${g}"' \
            -e WANDB_API_KEY=7db49f6a22d959fcba5d6bbaee1f7c28732dd745 \
            --name ${USER}_$(basename "$PWD" | tr '-' '_').gpu"${g}"  \
            $USER/$(basename "$PWD")"



eval "${COMMAND}"
