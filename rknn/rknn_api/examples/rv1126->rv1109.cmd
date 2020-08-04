echo 0 > /sys/devices/system/cpu/cpu2/online
echo 0 > /sys/devices/system/cpu/cpu3/online
echo 600000000 > /sys/devices/platform/ffbc0000.npu/devfreq/ffbc0000.npu/userspace/set_freq


echo 1 > /sys/devices/system/cpu/cpu2/online
echo 1 > /sys/devices/system/cpu/cpu3/online
echo 800000000 > /sys/devices/platform/ffbc0000.npu/devfreq/ffbc0000.npu/userspace/set_freq
