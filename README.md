# PointASNL_temp

pointnet pytorch ver. 과 호환되는 pointasnl 코드(임시).

---

사용법 
pointnet_pointnet2_pytorch/model에 위 code를 다운받아 사용.
train_classification.py 구동 시 model argument로 pointasnl_cls를 호출.

---

현재 개선 방향 

kd-tree(scipy 기반)을 gpu 기반으로 연산하는 과정이 필요. CuPy 등으로 시도해 볼 것.
accuracy : 90.4 / 39 epoch (still running). 원본 코드(tensorflow ver) : 90.8 / 39 epoch
