## build

```
mkdir build && cd build
cmake ..
make
make install
```

## install

```
adb push install/rknn_alternative_demo /userdata/rknn_alternative_demo
```

## run
```
adb shell
cd /userdata/rknn_alternative_demo/
./rknn_alternative_demo mobilenet_v1.rknn dog_224x224.jpg 224 224
```
