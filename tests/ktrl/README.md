# KTRL 离线回归

初始化仓库锁定的 SGLang 子模块后，在根目录运行 `python3 -B -m unittest discover -s tests/ktrl -p 'test_*.py'`。离线测试覆盖配置、共用专家放置、数据迭代、GRPO 梯度和证据解析。Graph replay 用例检查合成记录能否通过验证。真实 Graph 回放、专家反向和权重发布以 GPU 运行产物为准。
