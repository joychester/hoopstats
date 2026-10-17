# models/

Model files go here. They are not in the repository (too big, and each has its own licence); download them with:

```bash
cd ~/hoopstats
curl -L -o models/yolox_s.onnx https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx
curl -L -o models/yolox_m.onnx https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.onnx
# only needed for minutes (without --no-minutes)
curl -L -o osnet.zip https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/osnet/releases/v0.62.2/osnet-onnx-float.zip
unzip osnet.zip && mv osnet-onnx-float/osnet.onnx osnet-onnx-float/osnet.data models/
```

When done, this folder holds:

```
models/
  yolox_s.onnx    person and ball detection, default (YOLOX, Apache-2.0)
  yolox_m.onnx    larger detector, used with --accurate
  osnet.onnx      appearance features for minutes (OSNet, MIT)
  osnet.data
```
