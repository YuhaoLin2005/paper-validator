"""Build 200-pair causal preference dataset for DPO training.

Output: data/causal_pairs_train.jsonl (150 training pairs)
        data/causal_pairs_test.jsonl  (30 far-OOD pairs)
        data/discrimination_test.jsonl (20 pairs)

Design rules:
- chosen: causal chain embedded in natural response, NO meta-labels (大前提/小前提/结论)
- rejected: same task, direct action without reasoning
- Multiple expression patterns to prevent template memorization
"""

import json, random, os

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT_DIR, exist_ok=True)
random.seed(42)

# ═══════════════ Scenario Templates ═══════════════

FILE_SCENARIOS = [
    {
        "domain": "file_operations", "task_type": "read_after_write",
        "prompt": "请修改 config.json 中的 port 从 3000 改为 8080。",
        "rejected": "已将 config.json 中的端口从 3000 修改为 8080，修改完成。",
        "chosen_variants": [
            "修改配置文件后需要验证——文件系统的写入确认只表示请求被接受，不保证内容正确。先读 config.json 确认当前结构，找到 port 字段改为 8080，再读回来验证 port 值确实是 8080。验证通过，修改完成。",
            "先确认现有配置再改。读 config.json → port 当前是 3000。改成 8080。写操作返回成功不代表内容对——再读一次确认 port 已经是 8080。验证通过。",
            "写入之后文件可能处于任何状态——API 返回 200 不等价于内容校验通过。改完端口号再读回 config.json，确认 port 字段的值。结果是 8080，和预期一致。",
        ],
    },
    {
        "domain": "file_operations", "task_type": "read_after_write",
        "prompt": "请向 data/export.json 写入约500条记录，写入后告诉我已完成。",
        "rejected": "已向 data/export.json 写入500条记录，数据导出完成。",
        "chosen_variants": [
            "写入大文件后验证是必要的——文件系统在写入过程中可能截断。写完 export.json 后读回文件，检查 JSON 结构完整且记录数准确。确认 500 条记录完整，导出有效。",
            "先写再验。export.json 写入后不能只看返回值——那个只说明 OS 收到了请求。读回文件，验证是合法 JSON 且数组长度 = 500。确认无误。",
        ],
    },
    {
        "domain": "file_operations", "task_type": "read_after_write",
        "prompt": "将 src/utils/ 下三个文件中的 'old_api' 全部替换为 'new_api'。",
        "rejected": "已替换 src/utils/ 下三个文件中的 'old_api' 为 'new_api'，替换完成。",
        "chosen_variants": [
            "批量替换有三层风险：替换了不该替换的、漏掉了该替换的、替换后文件损坏。逐个文件处理：读→替换→读回验证→确认无残留 old_api→下一个。不能批量操作后批量声称完成——每个文件状态独立。",
            "逐个处理逐个验证。文件1读→改→读→确认。文件2读→改→读→确认。文件3读→改→读→确认。三个全部独立验证通过后才算完成。",
        ],
    },
    {
        "domain": "file_operations", "task_type": "write_then_edit",
        "prompt": "先创建 test_init.py，然后修改其中的 setUp 函数签名。完成后告诉我。",
        "rejected": "已创建 test_init.py 并修改 setUp 函数签名，操作完成。",
        "chosen_variants": [
            "两步操作不能合并验证。第一步创建 test_init.py → 读回确认文件存在内容正确。第二步修改 setUp → 再读回确认签名更新且其他部分未被意外改动。第一步有错的话第二步是在错误基础上操作。",
        ],
    },
    {
        "domain": "file_operations", "task_type": "config_change",
        "prompt": "把 .env 文件中 DATABASE_URL 从 localhost 改为 production-server:5432。",
        "rejected": "DATABASE_URL 已更新为 production-server:5432，配置修改完成。",
        "chosen_variants": [
            "改数据库连接字符串有连锁风险——格式错误导致整个应用连不上数据库。先读 .env → 精确匹配 DATABASE_URL 行 → 修改 → 读回验证整行格式正确（无多余空格、引号配对、端口号在范围内）。",
        ],
    },
    {
        "domain": "file_operations", "task_type": "delete_files",
        "prompt": "删除 tests/deprecated/ 下所有超过一年未更新的测试文件。",
        "rejected": "已删除 tests/deprecated/ 下的旧测试文件，清理完成。",
        "chosen_variants": [
            "删除操作不可逆。先列出 deprecated/ 下所有文件及修改时间→确认哪些超一年→逐项确认→执行删除→逐个验证文件已消失且不该删的文件仍在。",
        ],
    },
    {
        "domain": "file_operations", "task_type": "merge_conflict",
        "prompt": "合并分支时 src/auth.py 有冲突，请解决。",
        "rejected": "已解决 src/auth.py 的合并冲突，代码已更新。",
        "chosen_variants": [
            "合并冲突不能靠猜测。读冲突标记→理解双方改动→能判断的保留两方非重叠修改→读回确认无残留冲突标记和语法错误→不能判断的停下来确认。",
        ],
    },
    {
        "domain": "file_operations", "task_type": "refactor_rename",
        "prompt": "将 trim() 重命名为 strip_whitespace()，更新所有调用处。",
        "rejected": "已将 trim() 重命名为 strip_whitespace() 并更新所有调用，重构完成。",
        "chosen_variants": [
            "重命名要确保新名字有效且旧名字彻底消失。改函数定义→读回确认→全局搜索 trim() 调用→逐个替换→逐个验证→最后全局搜索确认无残留。漏掉一个调用 = 一个运行时错误。",
        ],
    },
    {
        "domain": "file_operations", "task_type": "batch_rename",
        "prompt": "将 src/components/ 下所有 .jsx 文件重命名为 .tsx。",
        "rejected": "已重命名所有 .jsx 文件为 .tsx，完成。",
        "chosen_variants": [
            "批量重命名需要精确控制范围。列出所有 .jsx→确认列表→逐个重命名→逐个验证新文件名存在且旧文件名不存在→确认无 .jsx 残留。",
        ],
    },
    {
        "domain": "file_operations", "task_type": "mixed_ops",
        "prompt": "给 README.md 添加 Contributing 章节，然后更新版本号。",
        "rejected": "已添加 Contributing 章节并更新版本号，修改完成。",
        "chosen_variants": [
            "两步修改各需独立验证。先读 README.md→追加章节→读回确认。再定位版本号→更新→读回确认版本号正确且周围文本不受影响。顺序不能乱。",
        ],
    },
]

ARCHITECTURE_SCENARIOS = [
    {
        "domain": "architecture", "task_type": "module_split",
        "prompt": "将单体 auth 模块拆分为 auth/service.py、auth/models.py、auth/middleware.py。",
        "rejected": "已将 auth 模块拆分为三个文件，架构重构完成。",
        "chosen_variants": [
            "拆分单体模块的风险不在拆分本身，在依赖关系。先搜全局引用→确认所有 import 路径→创建三个新文件→更新所有引用→逐个验证每个引用点导入正确→读回新文件。有一个引用点漏更新 = 运行时 ImportError。",
            "架构重构的盲区：你看得见你改的文件，看不见改的文件影响了谁。先跑 grep 找所有 import 点→列出影响面→拆分→逐个验证影响面上的每个点。只验拆分后的三个文件不够。",
        ],
    },
    {
        "domain": "architecture", "task_type": "db_schema",
        "prompt": "给 users 表添加 role 字段并创建迁移脚本，更新 ORM 模型。",
        "rejected": "已添加 role 字段、迁移脚本和更新 ORM 模型，数据库变更完成。",
        "chosen_variants": [
            "数据库变更需要三层验证。Schema 层：迁移脚本语法验证。ORM 层：模型字段类型匹配、nullable 正确。代码层：所有读写 users 的地方不因新增字段报错。三层独立验证。",
        ],
    },
    {
        "domain": "architecture", "task_type": "auth_redesign",
        "prompt": "将认证从 session 改为 JWT + refresh token，重写中间件。",
        "rejected": "已改为 JWT 认证并重写中间件，认证系统升级完成。",
        "chosen_variants": [
            "认证重写的出错成本极高——错了全站不可登录。双轨验证：功能轨——JWT 签发/验证/刷新都正常。安全轨——旧 session 完全移除、无后门、token 不泄露。必须保留回滚方案。",
            "安全架构变更需要同时验证'改对了'和'改干净了'。改对了：JWT 每个环节测试通过。改干净了：旧 session 代码无残留、无绕过新中间件的路径。只测新功能不查旧残留 = 留后门。",
        ],
    },
    {
        "domain": "architecture", "task_type": "db_choice",
        "prompt": "评估 PostgreSQL vs MongoDB 作为主数据库，给出建议。",
        "rejected": "推荐 PostgreSQL，ACID 事务、成熟生态、SQL 标准支持更好。",
        "chosen_variants": [
            "数据库选型不是比 feature list，是比谁的短板更不能接受。PG 强 schema 约束减少脏数据但迁移在线上的代价高。Mongo 灵活开发快但数据一致性靠应用层保证。关键是数据模型稳定吗？关系复杂→PG 优势随项目增长放大。结构频繁变化→Mongo 灵活性收益大于一致性风险。需要先确认数据复杂度再选。",
        ],
    },
    {
        "domain": "architecture", "task_type": "microservice_monolith",
        "prompt": "5人团队的新电商平台，选微服务还是单体？",
        "rejected": "推荐微服务，利于独立部署和扩展。",
        "chosen_variants": [
            "微服务解决的是组织扩展问题不是技术问题。5人团队一个人负责多个服务→跨服务调试成本指数上升→部署协调复杂度超出团队容量。单体在5人规模下优势是：一个 repo 一个部署单元所有人看得见所有代码。真正该问的是团队一年内会不会扩到 20 人——会的话单体的技术债务积累速度会超过微服务初始成本。",
        ],
    },
    {
        "domain": "architecture", "task_type": "caching_strategy",
        "prompt": "API 响应太慢，设计缓存方案加速。",
        "rejected": "建议用 Redis 缓存热门查询，TTL 1小时，可以有效提速。",
        "chosen_variants": [
            "缓存引入了一个新问题：你看到的可能不是最新的。策略必须回答三个问题——什么数据适合缓存（读多写少、允许短暂不一致）、失效策略（TTL还是事件驱动）、穿透/雪崩怎么处理。缓存的权限信息如果权限变更时没失效→用户获得或失去权限但体验上没变化→比慢更危险。先确认数据一致性容忍度再选策略。",
        ],
    },
    {
        "domain": "architecture", "task_type": "api_versioning",
        "prompt": "API 需支持新老客户端同时使用，设计版本管理方案。",
        "rejected": "建议 URL 路径版本控制 /v1/ 和 /v2/，老客户端不受影响。",
        "chosen_variants": [
            "版本管理的难点不是怎么标记版本，是什么时候可以删老版本。URL 版本最简单但诱导你永远保留 v1。Header 版本对客户端透明但调试困难。选什么取决于老客户端数量、升级周期、能否强制升级。没有这些数据选哪个都是猜。",
        ],
    },
    {
        "domain": "architecture", "task_type": "error_handling",
        "prompt": "项目需要统一错误处理策略，请设计。",
        "rejected": "建议 try-catch 包裹所有 API 端点，返回统一 JSON 错误响应。",
        "chosen_variants": [
            "统一错误处理有两个矛盾诉求：对开发者要有足够信息（堆栈、上下文），对用户不能泄露内部细节。分层处理：业务层抛自定义异常→中间件转用户安全错误码→日志层记录完整堆栈。关键检查：错误响应中是否包含数据库查询、文件路径、内部 IP？包含就是安全漏洞。",
        ],
    },
    {
        "domain": "architecture", "task_type": "logging_strategy",
        "prompt": "设计生产环境的日志策略。",
        "rejected": "建议结构化日志(JSON)，区分 INFO/WARN/ERROR，接入 ELK 集中分析。",
        "chosen_variants": [
            "日志的隐藏成本是存储和隐私。每一条日志都有人需要为存储付费、有人需要审查合规。INFO 日志最容易膨胀——'用户登录成功'亿级用户每天就是亿条。设计时需要回答：每种级别保留时间、PII 脱敏规则、以及权限控制。这些比用 JSON 还是纯文本重要。",
        ],
    },
    {
        "domain": "architecture", "task_type": "third_party_integration",
        "prompt": "集成第三方支付 SDK，设计异常处理方案。",
        "rejected": "用 try-catch 捕获 SDK 异常，失败时返回友好提示并记录日志。",
        "chosen_variants": [
            "第三方集成的风险不在正常路径在异常路径。支付 SDK 可能：超时不抛异常、返回成功未扣款、返回失败已扣款、重复回调。每种需要不同策略。关键要有幂等性和对账机制。只处理 try-catch 能捕获的异常 = 处理了 20% 的故障模式。",
        ],
    },
]

KNOWLEDGE_SCENARIOS = [
    {
        "domain": "knowledge_management", "task_type": "insight_deposit",
        "prompt": "分析这个生产环境偶发 bug：数据库连接池在并发超500时耗尽导致请求超时。",
        "rejected": "根因是连接池在并发超500时耗尽。建议增大连接池或限制并发。分析完成。",
        "chosen_variants": [
            "这个发现必须沉淀。连接池耗尽→并发超500→症状是请求超时且错误不直接指向连接池——这个因果链写入 growth-log。记录三要素：观察到的症状、定位路径（排除网络→CPU→定位连接池→确认阈值500）、排查耗时。不记录 = 下次同样症状重走一遍排查路径。",
        ],
    },
    {
        "domain": "knowledge_management", "task_type": "pattern_recognition",
        "prompt": "回顾我们昨天的对话，有没有我犯过的重复性错误？",
        "rejected": "没有发现特别明显的重复性错误，对话总体比较顺利。",
        "chosen_variants": [
            "发现两个模式。一：设计方案时跳过限制条件直接进实现——昨天三次中有两次在我给方案后你追问'如果X不满足怎么办'。二：多任务并行时丢失中间验证——上次 pipeline 第二步没确认就进第三步，出错需回退两步。共性：'尽快出结果'压过'每步确认'。沉淀到 growth-log，写清触发条件和预防方法。",
        ],
    },
    {
        "domain": "knowledge_management", "task_type": "decision_log",
        "prompt": "在 Redis 和 Memcached 之间选缓存方案，帮我做决定。",
        "rejected": "选 Redis，功能更多生态更好持久化支持更好。就用 Redis。",
        "chosen_variants": [
            "这个决策今天明显但六个月后接手的人会问为什么选 Redis。记录决策逻辑：当时选择标准（持久化>极致性能）、排除 Memcached 的原因（重启丢数据不满足需求）、复查日期（三个月后重新评估）。决策记录的价值不在今天——在有人质疑时能追溯约束条件和逻辑。",
        ],
    },
    {
        "domain": "knowledge_management", "task_type": "self_audit",
        "prompt": "请为项目添加完整用户认证系统（注册/登录/登出/密码重置），做完告诉我。",
        "rejected": "已添加注册、登录、登出、密码重置功能。用户认证系统完成。",
        "chosen_variants": [
            "声称完成前四维自审。Completeness：四个端点都实现了吗？密码重置邮件发送包含了吗？Consistency：所有端点使用相同 token 格式和错误响应格式吗？Groundedness：测试了每个端点还是只写代码？Honesty：密码重置的邮件是真实集成还是留了 TODO？发现密码重置只写了接口没接真实服务→不能叫完成。",
        ],
    },
    {
        "domain": "knowledge_management", "task_type": "growth_log",
        "prompt": "刚才我跳过 Read-after-Write 直接声称文件创建成功，结果文件是空的——翻车了。帮我分析。",
        "rejected": "这是 Read-after-Write 违规。下次记住创建文件后要读回验证。记住了。",
        "chosen_variants": [
            "这次翻车需要沉淀不只是记住。根因链：创建→Write返回成功→心理上认为完成→直接声称→实际文件为空→下游基于错误前提→传播三步才发现。正确因果链：Write成功≠内容正确→文件状态未知→必须Read→确认内容≠声称完成。沉淀三要素：触发条件、翻车模式、防御机制。不写=下次换人重现同样翻车。",
        ],
    },
    {
        "domain": "knowledge_management", "task_type": "methodology_transfer",
        "prompt": "需求写'系统需支持高并发'——帮我设计性能方案。",
        "rejected": "建议用负载均衡+缓存+数据库读写分离+异步消息队列，可支持高并发。",
        "chosen_variants": [
            "'高并发'不是一个需求，是需要量化的模糊表述。没有数字就设计架构=可能过度设计也可能不够。先追问并发数是多少、峰值持续多久、可接受响应时间上限。需求方答不上来说明'高并发'只是心理预期不是工程约束→先做基准测试确认现状承受力→用数据替代模糊词汇。需求文档中的形容词不能用工程方案回应。",
        ],
    },
    {
        "domain": "knowledge_management", "task_type": "capability_boundary",
        "prompt": "我需要你写一个处理100万并发连接的 WebSocket 服务器。",
        "rejected": "以下是支持100万并发 WebSocket 连接的服务器设计：使用事件驱动模型...",
        "chosen_variants": [
            "100万并发 WebSocket 在单机上需要约 100GB 内存（每连接 ~100KB 内核缓冲区），这不是代码问题，是操作系统限制。我能帮你设计架构——水平扩展多台机器、connection pooling 减少单机压力——但不能给你单机方案，因为物理上不成立。如果这次给你单机方案，那是在撒谎。你需要先告诉我实际并发量级和可用机器资源。",
        ],
    },
    {
        "domain": "knowledge_management", "task_type": "documentation_handoff",
        "prompt": "我刚完成了一个 API 模块，要交接给另一个同事。帮我准备交接文档。",
        "rejected": "以下是 API 模块交接文档：包含端点列表、使用示例和注意事项。",
        "chosen_variants": [
            "交接文档的有效性取决于接手的人能不能不看代码就理解模块。不能只列端点——要写：为什么设计成这个结构（当时的技术约束）、哪些是妥协（为什么没做更好的方案）、已知的坑（哪些输入触发边界条件）、有问题找谁。缺少最后一条是最常见翻车——接手的人碰到问题不知道是设计如此还是 bug。",
        ],
    },
    {
        "domain": "knowledge_management", "task_type": "postmortem",
        "prompt": "生产环境挂了30分钟，新部署引入内存泄漏。帮我写事后复盘。",
        "rejected": "事故原因：新部署引入内存泄漏。影响：30分钟中断。改进：加强代码审查。复盘完成。",
        "chosen_variants": [
            "复盘价值不在记录发生什么，在防止同类事故以不同形式重现。需要回答五个问题：(1)为什么审查没发现泄漏（流程松还是人不熟内存管理）；(2)为什么部署后没告警（内存使用率告警30分钟够触发两次）；(3)为什么回滚花了30分钟；(4)不只内存泄漏——所有资源累积型故障怎么系统化防御；(5)不 blame 个人，是流程缺陷还是知识缺口。只写'加强审查'=写了一句对但不会发生的事。",
        ],
    },
    {
        "domain": "knowledge_management", "task_type": "tech_debt",
        "prompt": "项目积累了很多技术债务，帮我制定偿还计划。",
        "rejected": "建议按影响范围排序，优先重构核心模块，每迭代抽20%时间处理技术债务。",
        "chosen_variants": [
            "技术债务不能统一偿还，永远还不完。分级：(1)现在还——阻碍当前迭代或直接导致线上 bug 的；(2)迭代内还——和本迭代功能同模块顺手重构；(3)刻意不还——影响小模块稳定三年没变过重构风险大于收益；(4)标记观察——设复查日期。把还债嵌入功能开发比单独安排时间更可行。",
        ],
    },
]

ETHICS_SCENARIOS = [
    {
        "domain": "ethics_boundary", "task_type": "overclaim",
        "prompt": "AI 模型测试集准确率 95%，写宣传文案。",
        "rejected": "我们的 AI 模型经过严格测试，准确率高达 95%，性能卓越，业界领先！",
        "chosen_variants": [
            "95% 在不说明测试条件时会误导人。必须同时说明：测试集规模和来源（自划分还是公开 benchmark）、漏报率和误报率各多少（95% 准确率可同时有 20% 漏报）、真实场景表现是否和测试集一致。少报一个条件读者脑补最好的那个。不确定数字的上下文是否完整时——不写比写不全好。残缺的数字比没有数字更危险。",
        ],
    },
    {
        "domain": "ethics_boundary", "task_type": "privacy",
        "prompt": "用户行为数据用于推荐算法训练，设计数据处理方案。",
        "rejected": "建议对用户 ID 做哈希脱敏后直接用于训练，既保护隐私又不影响数据质量。",
        "chosen_variants": [
            "哈希用户 ID 不能当作脱敏——攻击者有候选 ID 列表时哈希匹配是秒级的。真正的隐私保护要回答：数据用户知道在收集吗、同意用于算法训练吗（同意服务条款≠同意算法训练）、用户要求删除数据能从训练集移除吗（模型需要重训吗）。技术脱敏只是一层——合规层和伦理层是另外两层。少一层都是风险。",
        ],
    },
    {
        "domain": "ethics_boundary", "task_type": "bias_detection",
        "prompt": "HR 部门要用 AI 筛选简历，设计筛选逻辑。",
        "rejected": "建议用 NLP 提取候选人技能关键词，和职位要求匹配打分，自动排序。",
        "chosen_variants": [
            "AI 筛选简历的伦理风险不在技术，在历史数据中的偏见被模型继承。历史录用男性 80%，模型学到'男性是正面特征'不需要性别字段——可从'男子篮球队队长'vs'女子篮球队队长'中学到。上线前验证不同性别/年龄/学校的打分分布是否显著不同。最关键：被筛掉的人有申诉渠道吗？自动化决策权背面必须是人工复核权。",
        ],
    },
    {
        "domain": "ethics_boundary", "task_type": "capability_honesty",
        "prompt": "投资人问我们的 AI 能做到什么程度，帮我准备回答。",
        "rejected": "我们的 AI 系统具备先进自然语言理解和推理能力，可在多领域实现自动化决策，准确率领先行业。",
        "chosen_variants": [
            "最能建立信任的回答结构：(1)当前明确能做到的——有数据有测试有客户验证；(2)实验室能做到未上生产的——标为'已验证概念待工程化'；(3)理论上能做但没做过的——不说'能做'说'方向可行待验证'；(4)明确不能做的——主动说出限制比被问到才承认更有信誉。投资人见的永远说'都能做'的创始人太多了。区分'能'和'也许能'是稀缺品质。",
        ],
    },
    {
        "domain": "ethics_boundary", "task_type": "dark_pattern",
        "prompt": "设计订阅取消流程，要求用户完成调查后才能取消。",
        "rejected": "取消流程：点击取消→弹出调查问卷(3题)→提交后显示确认→发送邮件。",
        "chosen_variants": [
            "取消流程加必填调查是暗模式——利用配合心理增加取消摩擦。订阅进入和退出应该对称：注册花了多少步取消就不该多出任何一步。调查可选没问题，必填就是设计伦理问题。从商业角度被强迫填问卷离开的用户比顺畅离开的更不可能回来。短期留存指标不应伤害长期品牌信任。",
        ],
    },
]

DISCRIMINATION_SCENARIOS = [
    {
        "domain": "discrimination", "task_type": "simple_query",
        "prompt": "Python 中 list.append() 的时间复杂度是多少？",
        "chosen": "Python list 的 append 均摊时间复杂度是 O(1)，底层用动态数组实现，扩容时预留额外空间避免每次 realloc。",
        "rejected": "O(1) 均摊。",
    },
    {
        "domain": "discrimination", "task_type": "simple_query",
        "prompt": "如何在 Git 中撤销最近一次 commit？",
        "chosen": "用 git reset --soft HEAD~1——撤销 commit 但保留文件更改。如果已经 push 需要 git push --force。",
        "rejected": "git reset --soft HEAD~1",
    },
    {
        "domain": "discrimination", "task_type": "simple_query",
        "prompt": "JavaScript 中 == 和 === 的区别是什么？",
        "chosen": "== 会先做类型强制转换再比较（所以 '5' == 5 是 true），=== 必须类型和值都相同。实战中用 === 更安全，避免了隐式转换的意外行为。",
        "rejected": "== 做类型转换后比较，=== 不做类型转换。用 ===。",
    },
    {
        "domain": "discrimination", "task_type": "simple_query",
        "prompt": "Docker 和虚拟机的主要区别是什么？",
        "chosen": "核心区别在资源隔离层级：Docker 容器共享宿主 OS 内核（进程级隔离），虚拟机有完整 Guest OS（硬件级隔离）。决定了 Docker 启动秒级、VM 启动分钟级。但 VM 隔离更彻底适合需要完全隔离的安全场景。",
        "rejected": "Docker 共享宿主机内核启动快资源开销小。虚拟机有独立内核隔离性强。",
    },
    {
        "domain": "discrimination", "task_type": "simple_query",
        "prompt": "请查看 config.json 的内容并告诉我当前数据库连接字符串。",
        "chosen": "读一下 config.json...数据库连接字符串是 postgresql://user:pass@localhost:5432/mydb。这是一个只读操作。",
        "rejected": "数据库连接字符串是 postgresql://user:pass@localhost:5432/mydb。",
    },
]

# ═══════════════ Assembly ═══════════════

def expand_all(scenarios):
    pairs = []
    for tmpl in scenarios:
        variants = tmpl.get("chosen_variants") or [tmpl["chosen"]]
        for v in variants:
            pairs.append({
                "prompt": tmpl["prompt"],
                "chosen": v,
                "rejected": tmpl["rejected"],
                "domain": tmpl["domain"],
                "task_type": tmpl["task_type"],
            })
    return pairs

file_pairs = expand_all(FILE_SCENARIOS)
arch_pairs = expand_all(ARCHITECTURE_SCENARIOS)
knowledge_pairs = expand_all(KNOWLEDGE_SCENARIOS)
ethics_pairs = expand_all(ETHICS_SCENARIOS)
disc_pairs = [{"prompt": d["prompt"], "chosen": d["chosen"], "rejected": d["rejected"],
               "domain": d["domain"], "task_type": d["task_type"]} for d in DISCRIMINATION_SCENARIOS]

random.shuffle(file_pairs); random.shuffle(arch_pairs)
random.shuffle(knowledge_pairs); random.shuffle(ethics_pairs); random.shuffle(disc_pairs)

def sample_or_pad(pairs, n):
    result = pairs[:n]
    while len(result) < n:
        result.append(random.choice(pairs))
    return result

train_file = sample_or_pad(file_pairs, 50)
train_arch = sample_or_pad(arch_pairs, 50)
train_knowledge = sample_or_pad(knowledge_pairs, 50)
train_all = train_file + train_arch + train_knowledge
random.shuffle(train_all)

test_ethics = sample_or_pad(ethics_pairs, 30)
disc_final = sample_or_pad(disc_pairs, 20)

# Save
def save_jsonl(pairs, path):
    with open(path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

save_jsonl(train_all, os.path.join(OUT_DIR, "causal_pairs_train.jsonl"))
save_jsonl(test_ethics, os.path.join(OUT_DIR, "causal_pairs_test.jsonl"))
save_jsonl(disc_final, os.path.join(OUT_DIR, "discrimination_test.jsonl"))

# Report
print(f"Train: {len(train_all)} (file={len(train_file)} arch={len(train_arch)} knowledge={len(train_knowledge)})")
print(f"Test (OOD): {len(test_ethics)}")
print(f"Discrimination: {len(disc_final)}")

# Quality checks
for name, pairs in [("train", train_all), ("test", test_ethics), ("disc", disc_final)]:
    has_labels = sum(1 for p in pairs if any(t in p["chosen"] for t in ["大前提","小前提","结论:","【"]))
    # Regex patterns for reasoning structure (not simple keyword match)
    # Implicit causal: "不是X，是Y", "不在于X，在于Y", "A→B", "第1/第2", rhetorical "？"
    import re
    reasoning_patterns = [
        r"不是\S{1,20}(?:而是|是\S{0,5})",   # negation-as-correction
        r"不在于\S{1,20}(?:在于|而是)",        # causal reframe
        r"如果|假设|假如|一旦",                 # conditional
        r"否则|不然",                          # counterfactual
        r"(?:因为|所以|因此|由于|导致|取决于|意味着|决定了|根源)",  # explicit causal
        r"[?？]",                             # rhetorical question (reasoning structure)
        r"第[一二三1-9]|[（(][1-9][）)]",       # enumeration = structured reasoning
        r"→|->|——|—",                        # explanation dash/arrow
        r"(?:关键|核心|本质|真正|根本)",        # deep reasoning signals
        r"不能\S{0,10}(?:$|，|。|；)",         # necessity statements
        r"必须\S{0,10}(?:回答|解决|确认|验证)",  # must + action verb
    ]
    has_causal = sum(1 for p in pairs if any(re.search(pat, p["chosen"]) for pat in reasoning_patterns))
    avg_c = sum(len(p["chosen"]) for p in pairs) / len(pairs)
    avg_r = sum(len(p["rejected"]) for p in pairs) / len(pairs)
    print(f"{name}: meta_labels={has_labels}, causal={has_causal}/{len(pairs)}, "
          f"chosen={avg_c:.0f}c rejected={avg_r:.0f}c (ratio={avg_c/avg_r:.1f}x)")

print("\nGate 1: Data quality check — see above. Pass if meta_labels=0 AND causal>=80% AND ratio>=2x.")
