#!/bin/bash

# build base
__image=lab41/keras-cuda3
docker build -t $__image .
