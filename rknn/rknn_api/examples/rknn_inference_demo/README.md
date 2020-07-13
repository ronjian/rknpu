## build

```
mkdir build && cd build
cmake ..
make
make install
```

## install

```
adb push install/rknn_inference_demo /userdata/rknn_inference_demo
```

## run
```
adb shell
cd /userdata/rknn_inference_demo/
./rknn_inference_demo mobilenet_v1.rknn dog_224x224.jpg
```
