# ASVD: Activation-aware Singular Value Decomposition for Compressing Large Language Models

Paple link [paper](https://arxiv.org/abs/2312.05821).

# Requirement
- python>=3.10
- pip install -r requirements.txt


测试将协方差矩阵作为S的ASVD效果。
协方差矩阵计算位于act_aware_utils.py
协方差矩阵的\alpha参数控制位于asvd.py
ASVD对k_proj与v_proj的分解操作位于modules/svd_linear.py


Examples:
```
python asvd.py --model_id="your_model_path" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method covariance_mean --kv_cache_ratio_target 0.7 --use_cache --compress_kv_cache --dump_huggingface_model
```

