@echo off
cd /d C:\Users\nengj\OneDrive\Desktop\VSLAM
build\vslam.exe --sequence data\dataset\sequences\00 --hybrid --xfeat models\xfeat.pt --lg models\lighterglue.pt
