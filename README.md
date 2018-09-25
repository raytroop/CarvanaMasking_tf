This toy project is based on CS230 TensorFlow codebase in vision
[cs230 codebase](https://github.com/cs230-stanford/cs230-code-examples)
<br>
UNet is based on [image_segmentation.ipynb](https://github.com/tensorflow/models/blob/master/samples/outreach/blogs/segmentation_blogpost/image_segmentation.ipynb)

```
$ python train.py
Creating the datasets...
Creating the model...
Starting training for 10 epoch(s)
Epoch 1/10
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 128/128 [01:32<00:00,  1.63it/s, loss=0.236]
- Train metrics: accuracy: 0.958 ; loss: 0.489 ; mean_iou: 0.888
- Eval metrics: accuracy: 0.786 ; loss: 1.438 ; mean_iou: 0.393
- Found new best accuracy, saving in experiments/base_model/best_weights/after-epoch-1
Epoch 2/10
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 128/128 [01:26<00:00,  2.94it/s, loss=0.099]
- Train metrics: accuracy: 0.993 ; loss: 0.158 ; mean_iou: 0.978
- Eval metrics: accuracy: 0.786 ; loss: 1.677 ; mean_iou: 0.393
- Found new best accuracy, saving in experiments/base_model/best_weights/after-epoch-2
Epoch 3/10
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 128/128 [01:26<00:00,  2.94it/s, loss=0.057]
- Train metrics: accuracy: 0.994 ; loss: 0.076 ; mean_iou: 0.984
- Eval metrics: accuracy: 0.833 ; loss: 1.313 ; mean_iou: 0.522
- Found new best accuracy, saving in experiments/base_model/best_weights/after-epoch-3
Epoch 4/10
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 128/128 [01:26<00:00,  2.94it/s, loss=0.042]
- Train metrics: accuracy: 0.992 ; loss: 0.068 ; mean_iou: 0.977
- Eval metrics: accuracy: 0.886 ; loss: 0.868 ; mean_iou: 0.753
- Found new best accuracy, saving in experiments/base_model/best_weights/after-epoch-4
Epoch 5/10
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 128/128 [01:26<00:00,  2.94it/s, loss=0.034]
- Train metrics: accuracy: 0.995 ; loss: 0.039 ; mean_iou: 0.986
- Eval metrics: accuracy: 0.990 ; loss: 0.066 ; mean_iou: 0.973
- Found new best accuracy, saving in experiments/base_model/best_weights/after-epoch-5
Epoch 6/10
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 128/128 [01:29<00:00,  2.79it/s, loss=0.027]
- Train metrics: accuracy: 0.996 ; loss: 0.032 ; mean_iou: 0.988
- Eval metrics: accuracy: 0.990 ; loss: 0.068 ; mean_iou: 0.971
Epoch 7/10
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 128/128 [01:29<00:00,  2.91it/s, loss=0.023]
- Train metrics: accuracy: 0.996 ; loss: 0.029 ; mean_iou: 0.988
- Eval metrics: accuracy: 0.994 ; loss: 0.040 ; mean_iou: 0.983
- Found new best accuracy, saving in experiments/base_model/best_weights/after-epoch-7
Epoch 8/10
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 128/128 [01:28<00:00,  2.89it/s, loss=0.021]
- Train metrics: accuracy: 0.996 ; loss: 0.024 ; mean_iou: 0.990
- Eval metrics: accuracy: 0.996 ; loss: 0.027 ; mean_iou: 0.988
- Found new best accuracy, saving in experiments/base_model/best_weights/after-epoch-8
Epoch 9/10
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 128/128 [01:27<00:00,  2.92it/s, loss=0.020]
- Train metrics: accuracy: 0.997 ; loss: 0.022 ; mean_iou: 0.990
- Eval metrics: accuracy: 0.996 ; loss: 0.024 ; mean_iou: 0.989
- Found new best accuracy, saving in experiments/base_model/best_weights/after-epoch-9
Epoch 10/10
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 128/128 [01:30<00:00,  2.88it/s, loss=0.018]
- Train metrics: accuracy: 0.997 ; loss: 0.020 ; mean_iou: 0.991
- Eval metrics: accuracy: 0.996 ; loss: 0.022 ; mean_iou: 0.990
- Found new best accuracy, saving in experiments/base_model/best_weights/after-epoch-10
```