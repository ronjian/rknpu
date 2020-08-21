## build

```
mkdir build && cd build
cmake ..
make
make install
```

## install

```
adb push install/rknn_ofa /userdata/rknn_ofa
```

## run
```
adb shell
cd /userdata/rknn_ofa/
./rknn_ofa mobilenet_v1.rknn dog_224x224.jpg
```
