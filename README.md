# PointASNL_temp

pointnet pytorch ver. 과 호환되는 pointasnl 코드(임시).



#### 사용법 

> pointnet_pointnet2_pytorch/model에 위 code를 다운받아 사용.
> 
> train_classification.py, test_classification 구동 시 model argument로 pointasnl_cls를 호출.
>
> ex)
~~~
### hyperparameters need to be presetted
python train_classification.py --model pointasnl_cls --use_normals --use_uniform_sample --log_dir pointasnl_cls
python test_classification.py --log_dir pointasnl_cls --use_normals --use_uniform_sample
~~~


#### 현재 개선 방향 

> 지나치게 training 연산속도가 느림(Artemis 기준 1 epoch당 15분).  kd-tree(scipy 기반)을 gpu 기반으로 연산하는 과정이 필요(우선순위 높음). CuPy를 적용시켜 볼 여지 있음.
>   코드 최적화 역시 필요(우선순위 낮음)
>   일부 tensor가 channel last 방식으로 삽입되는 현상 발견
> 
> accuracy : 90.4 / 39 epoch (still running).
> 
> 원본 코드(tensorflow ver) : 90.8% / 39 epoch | 93.4% / 251 epoch
