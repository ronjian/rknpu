## build

```
mkdir build && cd build
cmake ..
make
make install
```

## install

```
adb push install/rknn_inference_memory /userdata/rknn_inference_memory
```

## run
```
adb shell
cd /userdata/rknn_inference_memory/
./rknn_inference_memory mobilenet_v1.rknn dog_224x224.jpg 224 224
```
