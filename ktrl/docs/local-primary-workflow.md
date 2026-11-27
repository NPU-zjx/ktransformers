# 本机主仓库与服务器执行副本

本机唯一主仓库为 `D:\my_all_learning\KTransformers\KT_FORK`，GitHub origin 为 `https://github.com/NPU-zjx/ktransformers.git`，工作分支为 `kt-rl`。

1. 日常编辑、审阅 diff、commit、push 都在本机主仓库完成。服务器仅作为执行和验证副本。服务器紧急修改须先导出补丁回到本机审阅、验证并提交，然后重新同步。
2. 同步前记录本机与服务器完整 SHA、dirty diff、未跟踪源码、子模块 SHA 和已安装 wheel/.so 来源。服务器有独有修改时先归档并合并回本机，禁止直接覆盖。
3. 已提交版本使用 `git fetch origin` 后按指定 SHA 快进或切换执行分支。新实验只从已校验的版本启动。运行中的实验固定源代码快照，等待结束后更新该执行目录。
4. 未提交改动确需上机验证时，传输本机生成的补丁/源码归档并保存 SHA256 清单。报告注明 base SHA、patch SHA 和 dirty 状态，验收后由本机提交。
5. 一致性验收同时检查 HEAD、主仓库全部跟踪文件的 Git blob hash、运行相关未跟踪文件、子模块提交与补丁。相同 HEAD 不能证明 dirty 工作树相同。
6. `third_party/llama.cpp` 的 MXFP4 扩展由仓库 `kt-kernel/third_party_patches/llama.cpp` 和 CMake 自动应用。构建后 dirty 必须与该补丁系列匹配。可用 build-local copy 配置避免修改源子模块。编译产物以源码哈希和构建记录验证。
7. 模型、数据、日志、checkpoint 和大 trace 存在仓库外；机器路径放本地配方/环境配置。代码同步不清理实验结果或运行依赖。
8. 本机提交通过测试后 push 到 `origin/kt-rl`，再记录 GitHub、主仓库、各已部署服务器的 SHA 与核验时间。未同步的机器标记旧副本，禁止宣称全部一致。

服务器目录为 sapphire1/2 的 `/mnt/data/zjx/ktrl-fifth/ktransformers`、qjh005 的 `/raid/data/zjx/kt-baseline/ktransformers`。服务器历史分支和本地补丁先备份，正式运行目录以部署记录为准。
