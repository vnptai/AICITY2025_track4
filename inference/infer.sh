#!/bin/bash

cd yolo11/build

cmake ..
make -j

./yolo11_det -s ../yolo11m.wts yolo11m.engine m
./yolo11_det -d ../yolo11m.engine ../../../data/test/images c
