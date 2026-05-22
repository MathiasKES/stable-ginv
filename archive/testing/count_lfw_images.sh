#!/bin/bash
ROOT="/work3/s234843/bachelor/datasets/lfw/lfw-deepfunneled/lfw-deepfunneled"
echo "Identities: $(ls "$ROOT" | wc -l)"
echo "Images:     $(find "$ROOT" -name "*.jpg" | wc -l)"
