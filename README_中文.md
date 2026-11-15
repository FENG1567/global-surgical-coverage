# 全球手术地理覆盖项目复现仓库

对应2026年11月14日最终稿。本目录可作为GitHub仓库根目录上传，不需要上传父目录里的整理报告。

## 可以直接复现什么

- 用随附源表重新绘制6幅主图和4幅补图，不复制现成PDF冒充重绘。
- 导出主表1–4及补表S1–S23，核对主表2–4的下界、模型表现和遮蔽误差。
- 用市级汇总输入重新拟合三术式的分母模型，核对原10次空间分区结果。
- 检查设施选择预算、可行构成、记录年份审计和模型结果的版本对应关系。

入口命令见英文README。安装环境使用environment/requirements-validated.lock.txt，不要直接使用混有其他项目包的系统Python。

## 原始数据路线

原始SIH、CNES、SAEH/CLUES、OSM、GHSL、边界与摩擦栅格仍由原提供方获取。
本仓库提供来源清单、已知SHA-256、获取辅助脚本和科学代码；没有把第三方数据库改标成自己的CC BY 4.0。
完整的原始数据到全球栅格流程没有在本版本中全部重跑。少数继承的设施与登记提取版本标识仍不完整，见docs/FULL_ANALYSIS.md；这不妨碍随附结果的复画和市级模型重新拟合。

## 上传和授权

自编代码采用MIT，自有汇总结果和图采用CC BY 4.0。不要另传原始入院行、账号配置、虚拟环境或工作栅格。
公开GitHub仓库：https://github.com/FENG1567/global-surgical-coverage；引用版本v1.0.1。

## 图表展示与复现

当前绘图入口为 scripts/reproduce_figures.py，调用 code/analysis/plotting/layout/ 中的模块；早期绘图脚本作为历史源版本保留。make figures 输出PDF/SVG、600 dpi RGB LZW TIFF与300 dpi PNG。主表1为27国统计概况，完整逐国端点对应当前正文补表S4（旧分析源文件为data/tables/Table_S2.csv）；make tables 同时导出分析源CSV及与正文、补充材料对应的展示表。主表3将两组AUC分别以“基线→加入活动”并列显示，全精度结果仍保留。

编号展示表共27张；data/tables/另保留Main_Table_1_full_fields.csv作为主表1完整明细的辅助源数据，因此源CSV共28份。辅助源CSV正常导出，不重复生成编号展示表。


## 2026年11月15日图稿排版更新

10张参考PDF与复画代码同步采用大写Arial Bold 18 pt面板标注及加粗文字，原生宽183 mm。Windows检测已安装Arial；其他系统通过 --font-dir 指定合法安装字体所在目录，字体文件不随包分发。补图顺序跟随当前稿件：S1信息揭示，S2严格验证与墨西哥，S3巴西全国构成，S4州约束与验证。源数据数值不变。


图件读者复核：10图已明确差值单位、术式行名与分面图例归属，复画代码和参考PDF同步。源数据数值保持不变。


## 当前稿件表号

复表入口scripts/reproduce_tables.py按当前稿件S1–S23编号输出CSV与Markdown。data/tables中保留原分析文件名，避免改动既有分析路径。metadata/publication_table_crosswalk.csv逐表列出当前编号、原分析文件及不变的SHA-256。

## 代码可用性

自编分析与复现代码以MIT许可公开于https://github.com/FENG1567/global-surgical-coverage，版本v1.0.1。仓库含汇总输入、图表复现脚本、市级模型重新拟合代码、环境要求与源数据记录。作者汇总结果和图稿采用CC BY 4.0。完整空间分析的重建需要按原提供方条款另行获取外部数据清单中的地理和登记输入。

GitHub自动工作流核验发布文件、复现表格并运行回归测试。精确复画另需合法安装Arial Regular与Bold，按文档在本地运行绘图命令，不在CI中下载未授权字体。
