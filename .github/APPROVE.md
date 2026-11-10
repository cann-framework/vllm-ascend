# 评论批准与自动合并

在 PR 的普通评论区单独发送 `/approve`。本仓管理员可以批准所有路径及自己的 PR；有本仓写权限的 CODEOWNERS 可批准其负责的路径，但不能批准自己创建的 PR。涉及多个路径时，每个路径都须得到授权覆盖；重命名会检查新旧路径。这与 `cann-deepinfer-recipe` 禁止所有作者自批的规则不同。

批准绑定 PR、当前 Head、目标分支和 Base 提交以及当前 CODEOWNERS。评论被修改或删除、权限撤销、PR 版本变化、草稿和未解决的有效变更请求都会阻止合并。机器人还会核对该 PR 当前版本的 Smoke 状态、受信工作流和 run/attempt、完整计划及预合并源码 artifact、逐例 job；缺失或不匹配均不能合并。其他必需检查未通过时，本工作流将在检查完成或定时重评后重试；合并前重新检查权限、版本和 Smoke，不提前委托 GitHub 自动合并。

工作流仅读取默认分支上的控制代码及 CODEOWNERS，使用已有的 `lite-actions-torchtitan-checks` App 发布 `committer-approval` 检查，并由 Actions 机器人执行合并，不创建原生 Review。App 私钥仅放在只允许 main 的 `committer-approval` 环境；工作流、审批脚本和 CODEOWNERS 的更改只接受管理员批准。

## 启用与回滚

1. 将既有 Checks App 安装到本仓，权限限于 `Checks: write` 和元数据读取；配置仅允许 main 的 `committer-approval` 环境及其 secret `APPROVAL_APP_PRIVATE_KEY`，设置仓库变量 `APPROVAL_APP_ID=5575236`。
2. 默认设置 `APPROVAL_AUTOMATION_ENABLED=false`。Smoke 接口和本工作流到达默认分支后，先手动运行 `Committer approval` 的 `initialize` 模式，为所有已打开 PR 生成当前版本快照；初始化前的评论不计为批准。启用自动审批后立即手动运行 `reconcile` 模式，补齐初始化与启用之间新建或版本改变的 PR；后续也会定时补扫，已有同版本的审批不重置。本方案无需开启 Actions 创建和批准 PR 的仓库开关。
3. 参照 `cann-deepinfer-recipe`，在 PR 规则中要求来自该 Checks App 的 `committer-approval` 检查，将必需的原生批准次数设为 0，并允许管理员 bypass。保留 PR、对话解决、禁止强推和删除等原有保护；规则变更前保存原配置。
4. 在 main 和 release 的临时 PR 上验证管理员自批、CODEOWNER 授权、普通作者自批拒绝、评论撤回、新提交失效、检查等待后重试以及实际合并；生产 PR 未验证前不要用作试验。异常时先将启用变量设为 `false`，恢复保存的规则并检查未完成 PR 的状态。

Smoke 的 source 和计划 artifact 当前仅保留 7 天；过期、缺失或不可读取时会阻止合并。对版本和计划仍相同的旧运行选择“Re-run all jobs”会产生新 attempt；也可从 main 手动触发 `Dispatcher-Smoke`，填写 `source_pr`、当前 `expected_head_sha`、`expected_base_sha` 及 32 位十六进制 `bridge_nonce`，留空 `test_cases` 以生成完整计划。等待新的完整结果和 artifact，再重试审批；当前不会仅因 artifact 过期而自动重跑。

启用与补扫完成之间，未初始化 PR 的首次 `/approve` 仍可能要求重发；补扫和评论事件并发时，也可能要求对同一版本重新批准。旧评论不会因补扫被自动采纳。

## 权限边界

与参考仓相同，GitHub 必需检查绑定提交 SHA，不能单独区分两个使用相同 Head 的 PR；一个 PR 上的成功检查可能被另一 PR 复用。具有仓库 Write 权限的人也可能通过其他 Actions 工作流使用机器人权限。仓库选择接受这一权限边界，应只向可信成员授予 Write 权限。

GitHub 检查、权限复核与合并是独立 API，无法原子执行。工作流中断时检查状态可能滞留，后续事件或定时任务会重评；若任务无法运行，须人工核对 PR 和检查状态。
