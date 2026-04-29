const fs = await import("node:fs/promises");
const path = await import("node:path");
const { Presentation, PresentationFile } = await import("@oai/artifact-tool");

const ROOT = path.resolve(".");
const OUT_DIR = path.join(ROOT, "paper", "ppt_report");
const SCRATCH_DIR = path.join(ROOT, "tmp", "slides", "dp_report");
const PREVIEW_DIR = path.join(SCRATCH_DIR, "preview");
const INSPECT_PATH = path.join(SCRATCH_DIR, "inspect.ndjson");

const W = 1280;
const H = 720;

const C = {
  navy: "#0B2C5F",
  blue: "#1E5AA8",
  cyan: "#2F9EC8",
  green: "#22A06B",
  mint: "#E8F5EE",
  paleBlue: "#EAF2FC",
  ink: "#182235",
  muted: "#5F6D7A",
  light: "#F6F9FC",
  line: "#D8E2EA",
  white: "#FFFFFF",
  amber: "#D89827",
  red: "#D95A4E",
  transparent: "#00000000",
};

const FONT = {
  cn: "Microsoft YaHei",
  title: "Microsoft YaHei",
  mono: "Consolas",
};

const logoPath = path.join(ROOT, "tmp", "ppt_template_media", "image1.png");
const cifarImage = path.join(
  ROOT,
  "dp_experiments",
  "outputs",
  "next_priority_experiments",
  "repeats",
  "cifar10_eps1p5_k1000_cnn_seed42",
  "visuals",
  "cifar10",
  "diffusion_dp_eps_1.5_k_1000.png",
);
const mnistImage = path.join(
  ROOT,
  "dp_experiments",
  "outputs",
  "next_priority_experiments",
  "repeats",
  "mnist_eps0p5_k640_resnet20_seed42",
  "visuals",
  "mnist",
  "diffusion_dp_eps_0.5_k_640.png",
);

const inspect = [];

async function ensureDirs() {
  await fs.mkdir(OUT_DIR, { recursive: true });
  await fs.mkdir(PREVIEW_DIR, { recursive: true });
}

async function readImageBlob(imagePath) {
  const bytes = await fs.readFile(imagePath);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

function line(fill = C.line, width = 1) {
  return { style: "solid", fill, width };
}

function addShape(slide, slideNo, geometry, x, y, w, h, fill = C.white, stroke = C.transparent, width = 0, role = "shape") {
  const shape = slide.shapes.add({
    geometry,
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: line(stroke, width),
  });
  inspect.push({ kind: "shape", slide: slideNo, role, bbox: [x, y, w, h] });
  return shape;
}

function addText(slide, slideNo, text, x, y, w, h, opts = {}) {
  const shape = addShape(slide, slideNo, "rect", x, y, w, h, opts.fill ?? C.transparent, opts.stroke ?? C.transparent, opts.strokeWidth ?? 0, opts.role ?? "text");
  shape.text = text;
  shape.text.fontSize = opts.size ?? 22;
  shape.text.typeface = opts.face ?? FONT.cn;
  shape.text.color = opts.color ?? C.ink;
  shape.text.bold = Boolean(opts.bold);
  shape.text.alignment = opts.align ?? "left";
  shape.text.verticalAlignment = opts.valign ?? "top";
  shape.text.insets = opts.insets ?? { left: 0, right: 0, top: 0, bottom: 0 };
  if (opts.autoFit) shape.text.autoFit = opts.autoFit;
  inspect.push({
    kind: "textbox",
    slide: slideNo,
    role: opts.role ?? "text",
    text: String(text ?? ""),
    textChars: String(text ?? "").length,
    textLines: String(text ?? "").split(/\n/).length,
    bbox: [x, y, w, h],
  });
  return shape;
}

async function addImage(slide, slideNo, imagePath, x, y, w, h, fit = "contain", role = "image") {
  const image = slide.images.add({
    blob: await readImageBlob(imagePath),
    fit,
    alt: role,
  });
  image.position = { left: x, top: y, width: w, height: h };
  inspect.push({ kind: "image", slide: slideNo, role, path: imagePath, bbox: [x, y, w, h] });
  return image;
}

function addHeader(slide, slideNo, title, section = "差分隐私合成数据外包训练") {
  slide.background.fill = C.white;
  addShape(slide, slideNo, "rect", 0, 0, W, 18, C.navy, C.transparent, 0, "top identity bar");
  addShape(slide, slideNo, "rect", 0, 18, W, 4, C.green, C.transparent, 0, "top accent bar");
  addText(slide, slideNo, section, 64, 38, 440, 28, { size: 14, color: C.blue, bold: true, role: "section label" });
  addText(slide, slideNo, String(slideNo).padStart(2, "0"), 1156, 38, 60, 28, { size: 16, color: C.blue, bold: true, align: "right", role: "slide number" });
  addText(slide, slideNo, title, 64, 82, 850, 64, { size: 34, color: C.ink, bold: true, role: "title" });
  addShape(slide, slideNo, "rect", 64, 154, 1152, 1.6, C.line, C.transparent, 0, "title rule");
}

function addFooter(slide, slideNo, text = "数据来源：DP-private 实验记录与 dp_experiments/outputs") {
  addText(slide, slideNo, text, 64, 676, 900, 22, { size: 11, color: C.muted, role: "footer" });
}

function addBulletList(slide, slideNo, items, x, y, w, gap = 42, opts = {}) {
  items.forEach((item, i) => {
    const yy = y + i * gap;
    addShape(slide, slideNo, "ellipse", x, yy + 8, 10, 10, opts.dot ?? C.green, C.transparent, 0, "bullet dot");
    addText(slide, slideNo, item, x + 22, yy, w - 22, gap - 3, {
      size: opts.size ?? 20,
      color: opts.color ?? C.ink,
      role: "bullet",
      autoFit: "shrinkText",
    });
  });
}

function addMetric(slide, slideNo, x, y, w, h, value, label, accent = C.blue, note = "") {
  addShape(slide, slideNo, "roundRect", x, y, w, h, C.white, C.line, 1.2, "metric card");
  addShape(slide, slideNo, "rect", x, y, 8, h, accent, C.transparent, 0, "metric accent");
  addText(slide, slideNo, value, x + 24, y + 22, w - 48, 44, { size: 30, color: accent, bold: true, role: "metric value" });
  addText(slide, slideNo, label, x + 24, y + 74, w - 48, 44, { size: 16, color: C.ink, role: "metric label", autoFit: "shrinkText" });
  if (note) addText(slide, slideNo, note, x + 24, y + h - 34, w - 48, 20, { size: 10, color: C.muted, role: "metric note" });
}

function addCard(slide, slideNo, x, y, w, h, title, body, accent = C.blue) {
  addShape(slide, slideNo, "roundRect", x, y, w, h, C.light, C.line, 1, "content card");
  addShape(slide, slideNo, "rect", x, y, w, 6, accent, C.transparent, 0, "card accent");
  addText(slide, slideNo, title, x + 20, y + 18, w - 40, 28, { size: 17, color: accent, bold: true, role: "card title" });
  addText(slide, slideNo, body, x + 20, y + 58, w - 40, h - 76, { size: 15, color: C.ink, role: "card body", autoFit: "shrinkText" });
}

function addTable(slide, slideNo, x, y, colWidths, rowH, rows, opts = {}) {
  const totalW = colWidths.reduce((a, b) => a + b, 0);
  rows.forEach((row, r) => {
    let xx = x;
    const fill = r === 0 ? (opts.headerFill ?? C.navy) : r % 2 ? C.white : C.light;
    row.forEach((cell, c) => {
      const w = colWidths[c];
      addShape(slide, slideNo, "rect", xx, y + r * rowH, w, rowH, fill, C.line, 0.8, "table cell");
      addText(slide, slideNo, cell, xx + 8, y + r * rowH + 7, w - 16, rowH - 10, {
        size: r === 0 ? 13 : (opts.size ?? 13),
        color: r === 0 ? C.white : C.ink,
        bold: r === 0,
        role: "table text",
        autoFit: "shrinkText",
      });
      xx += w;
    });
  });
  addShape(slide, slideNo, "rect", x, y, totalW, rowH * rows.length, C.transparent, C.line, 1, "table outline");
}

function addMiniFormula(slide, slideNo, x, y, w, text, label = "") {
  addShape(slide, slideNo, "roundRect", x, y, w, 74, C.paleBlue, C.line, 1, "formula panel");
  if (label) addText(slide, slideNo, label, x + 18, y + 12, 150, 18, { size: 12, color: C.blue, bold: true, role: "formula label" });
  addText(slide, slideNo, text, x + 18, y + 34, w - 36, 30, { size: 18, color: C.ink, face: FONT.mono, role: "formula" });
}

function styleChart(chart) {
  chart.hasLegend = true;
  chart.legend.position = "bottom";
  chart.legend.textStyle.fontSize = 12;
  chart.legend.textStyle.typeface = FONT.cn;
  chart.xAxis.textStyle.fontSize = 11;
  chart.xAxis.textStyle.typeface = FONT.cn;
  chart.yAxis.textStyle.fontSize = 11;
  chart.yAxis.textStyle.typeface = FONT.cn;
  chart.plotAreaFill = C.white;
}

function addBarChart(slide, slideNo, x, y, w, h, title, categories, seriesDefs) {
  addShape(slide, slideNo, "roundRect", x, y, w, h, C.white, C.line, 1, "chart panel");
  const chart = slide.charts.add("bar");
  chart.position = { left: x + 18, top: y + 44, width: w - 36, height: h - 62 };
  chart.title = title;
  chart.titleTextStyle.fontSize = 15;
  chart.titleTextStyle.typeface = FONT.cn;
  chart.titleTextStyle.fill = C.ink;
  chart.categories = categories;
  chart.barOptions.direction = "column";
  chart.barOptions.grouping = "clustered";
  seriesDefs.forEach((def) => {
    const s = chart.series.add(def.name);
    s.values = def.values;
    s.categories = categories;
    s.fill = def.color;
    s.stroke = line(def.color, 1);
  });
  styleChart(chart);
  inspect.push({ kind: "chart", slide: slideNo, role: title, chartType: "bar", bbox: [x, y, w, h] });
  return chart;
}

function addLineChart(slide, slideNo, x, y, w, h, title, categories, seriesDefs) {
  addShape(slide, slideNo, "roundRect", x, y, w, h, C.white, C.line, 1, "chart panel");
  const chart = slide.charts.add("line");
  chart.position = { left: x + 18, top: y + 44, width: w - 36, height: h - 62 };
  chart.title = title;
  chart.titleTextStyle.fontSize = 15;
  chart.titleTextStyle.typeface = FONT.cn;
  chart.titleTextStyle.fill = C.ink;
  chart.categories = categories;
  chart.lineOptions.grouping = "standard";
  chart.lineOptions.smooth = false;
  seriesDefs.forEach((def) => {
    const s = chart.series.add(def.name);
    s.values = def.values;
    s.categories = categories;
    s.stroke = line(def.color, 2.4);
    s.fill = def.color;
  });
  styleChart(chart);
  inspect.push({ kind: "chart", slide: slideNo, role: title, chartType: "line", bbox: [x, y, w, h] });
  return chart;
}

async function slide1(p) {
  const s = p.slides.add();
  const n = 1;
  s.background.fill = C.white;
  addShape(s, n, "rect", 0, 0, 405, H, C.navy, C.transparent, 0, "cover side panel");
  addShape(s, n, "rect", 405, 0, 8, H, C.green, C.transparent, 0, "cover accent");
  await addImage(s, n, logoPath, 48, 42, 306, 58, "contain", "school logo");
  addText(s, n, "导师汇报", 64, 158, 180, 28, { size: 19, color: "#B7D8FF", bold: true, role: "kicker" });
  addText(s, n, "差分隐私合成数据\n外包训练实验流程", 64, 205, 300, 150, { size: 36, color: C.white, bold: true, role: "cover title" });
  addText(s, n, "原始图像不出本地，仅输出经隐私机制处理的合成数据；用原始验证集准确率判断外包训练可用性。", 64, 386, 290, 96, { size: 17, color: "#DCEBFA", role: "cover subtitle", autoFit: "shrinkText" });
  addMetric(s, n, 470, 154, 226, 142, "98.54%±0.14pp", "MNIST 最终 3 seeds original_acc", C.green);
  addMetric(s, n, 732, 154, 226, 142, "60.32%±1.44pp", "CIFAR10 最终 3 seeds original_acc", C.blue);
  addMetric(s, n, 994, 154, 226, 142, "+13.74pp", "CIFAR10 从 47.66% 提升至 61.40%", C.amber);
  addShape(s, n, "roundRect", 470, 350, 750, 210, C.light, C.line, 1, "headline panel");
  addText(s, n, "主结论", 500, 382, 100, 26, { size: 18, color: C.blue, bold: true, role: "headline label" });
  addText(s, n, "Diffusion+DP 是当前最值得保留的主方法：MNIST 已达到高准确率且波动很小；CIFAR10 的主要瓶颈转为合成样本质量与规模，而不是 epsilon=1.0 到 1.5 的细调。", 500, 424, 660, 86, { size: 22, color: C.ink, bold: true, role: "headline", autoFit: "shrinkText" });
  addFooter(s, n, "汇报依据：实验流程记录 425、complete_original_models、next_priority_experiments");
}

async function slide2(p) {
  const s = p.slides.add();
  const n = 2;
  addHeader(s, n, "研究问题：原始数据不出域，外包训练是否仍可用？");
  const steps = [
    ["企业本地原始数据", "MNIST / CIFAR10\n真实图像保留在本地"],
    ["本地隐私处理", "DP / GAN+DP / Distill+DP / Diffusion+DP"],
    ["第三方算力训练", "仅接收合成数据\n训练 CNN / ResNet20"],
    ["本地评估", "original_acc 作为主指标\nsynthetic_acc 作为辅助指标"],
  ];
  steps.forEach((st, i) => {
    const x = 78 + i * 294;
    addShape(s, n, "roundRect", x, 232, 230, 166, i === 1 ? C.mint : C.paleBlue, C.line, 1.2, "workflow step");
    addText(s, n, st[0], x + 18, 258, 194, 30, { size: 18, color: C.navy, bold: true, role: "workflow title" });
    addText(s, n, st[1], x + 18, 304, 194, 64, { size: 15, color: C.ink, role: "workflow body", autoFit: "shrinkText" });
    if (i < steps.length - 1) addShape(s, n, "rightArrow", x + 236, 288, 58, 48, C.green, C.transparent, 0, "workflow arrow");
  });
  addMiniFormula(s, n, 96, 466, 500, "Pr[M(D) ∈ S] ≤ e^ε Pr[M(D') ∈ S] + δ", "差分隐私约束");
  addMiniFormula(s, n, 654, 466, 500, "θ* = argminθ 1/|Dsyn| Σ ℓ(fθ(x), y)", "外包训练目标");
  addBulletList(s, n, [
    "核心不是只看合成数据上的拟合，而是看模型能否迁移到真实原始验证集。",
    "实验围绕隐私预算 ε、每类合成样本数 k、训练轮数和下游模型复杂度展开。",
  ], 90, 582, 1060, 38, { size: 17 });
  addFooter(s, n);
}

async function slide3(p) {
  const s = p.slides.add();
  const n = 3;
  addHeader(s, n, "实验矩阵与评价指标");
  addTable(s, n, 72, 190, [144, 258, 190, 190, 210, 150], 48, [
    ["维度", "取值", "说明", "主输出", "本轮变化", "位置"],
    ["数据集", "MNIST / CIFAR10", "灰度数字与彩色图像", "original_acc", "复跑完整矩阵", "metrics.csv"],
    ["方法", "DP / GAN+DP / Distill+DP / Diffusion+DP", "四类脱敏路线", "方法排序", "最终收敛到 Diffusion+DP", "method"],
    ["隐私预算", "ε=0.1/0.5/1.0；扩展 0.7/1.5/2.0", "ε 越大噪声约束越弱", "隐私-效用权衡", "局部扫参", "epsilon"],
    ["样本规模", "k=160/320/640/1000", "每类合成样本数", "样本规模趋势", "CIFAR10 扩到 1000", "per_class"],
    ["下游模型", "CNN / ResNet20", "保持结构不变", "训练与泛化", "不靠换模型提分", "classifier"],
  ], { size: 12 });
  addMiniFormula(s, n, 92, 520, 326, "Accorig = (1/N) Σ 1[fθ(xi)=yi]", "主指标");
  addMiniFormula(s, n, 476, 520, 326, "Gap = Accsyn − Accorig", "泛化差距");
  addMiniFormula(s, n, 860, 520, 326, "Dsyn = Mε,k(Dlocal)", "合成数据");
  addFooter(s, n);
}

async function slide4(p) {
  const s = p.slides.add();
  const n = 4;
  addHeader(s, n, "基础矩阵：Diffusion+DP 先胜出，GAN+DP 暂不主攻");
  addBarChart(s, n, 70, 194, 682, 354, "完整矩阵各方法 mean original_acc（%）", ["DP", "GAN+DP", "Distill+DP", "Diffusion+DP"], [
    { name: "MNIST", values: [80.67, 10.27, 50.57, 91.82], color: C.green },
    { name: "CIFAR10", values: [31.52, 9.74, 18.99, 40.66], color: C.blue },
  ]);
  addCard(s, n, 790, 190, 386, 96, "观察 1：方法排序稳定", "两类数据集上 Diffusion+DP 的平均 original_acc 均最高，说明路线值得集中计算资源继续扩展。", C.green);
  addCard(s, n, 790, 312, 386, 96, "观察 2：GAN+DP 低效", "当前轻量 GAN 设置下接近随机水平，生成耗时较高，后续不作为主攻方向。", C.red);
  addCard(s, n, 790, 434, 386, 114, "观察 3：synthetic_acc 需谨慎", "Distillation+DP 在合成域可能很高，但真实验证集不稳定，提示必须以 original_acc 为主。", C.amber);
  addFooter(s, n, "统计口径：complete_original_models/mnist 与 cifar10 的 48 行完整矩阵");
}

async function slide5(p) {
  const s = p.slides.add();
  const n = 5;
  addHeader(s, n, "评估方式修正：避免合成训练集自测虚高");
  addShape(s, n, "roundRect", 88, 208, 294, 128, C.paleBlue, C.line, 1, "split box");
  addText(s, n, "旧口径", 112, 232, 100, 28, { size: 19, color: C.red, bold: true, role: "old label" });
  addText(s, n, "合成样本既训练又测试\nsynthetic_acc 可能反映记忆", 112, 274, 230, 44, { size: 16, role: "old body", autoFit: "shrinkText" });
  addShape(s, n, "rightArrow", 416, 246, 80, 54, C.green, C.transparent, 0, "correction arrow");
  addShape(s, n, "roundRect", 532, 208, 560, 128, C.mint, C.line, 1, "split box");
  addText(s, n, "新口径：按类别 8:2 拆分", 560, 232, 300, 28, { size: 19, color: C.green, bold: true, role: "new label" });
  addText(s, n, "80% Dsyn_train 训练分类器；20% Dsyn_eval 计算 synthetic_acc；原始验证集计算 original_acc。", 560, 274, 480, 44, { size: 16, role: "new body", autoFit: "shrinkText" });
  addMiniFormula(s, n, 110, 408, 440, "Dsyn = Dtrain_syn ∪ Deval_syn,  |Deval|/|Dsyn| = 0.2", "拆分公式");
  addMiniFormula(s, n, 640, 408, 440, "Ranking = sort_by(original_acc)", "排序原则");
  addBulletList(s, n, [
    "修正后 synthetic_acc 是辅助指标，主要用于观察合成域内泛化。",
    "original_acc 代表模型在真实原始分布上的可用性，是汇报里的主指标。",
    "视觉质量统计同时裁掉标题和空白区域，使 SSIM/PSNR 更接近主体图像差异。",
  ], 124, 540, 996, 36, { size: 17 });
  addFooter(s, n, "对应代码：trainer.py、run_pipeline.py、generate_paper_outputs.py");
}

async function slide6(p) {
  const s = p.slides.add();
  const n = 6;
  addHeader(s, n, "完整矩阵复跑：MNIST 接近可用，CIFAR10 仍是难点");
  addMetric(s, n, 84, 190, 260, 136, "96.91%", "MNIST 最优：Diffusion+DP, ε=0.5, k=320, ResNet20", C.green);
  addMetric(s, n, 374, 190, 260, 136, "47.66%", "CIFAR10 最优：Diffusion+DP, ε=1.0, k=320, CNN", C.blue);
  addMetric(s, n, 664, 190, 260, 136, "91.82%", "MNIST Diffusion+DP 平均 original_acc", C.green);
  addMetric(s, n, 954, 190, 260, 136, "40.66%", "CIFAR10 Diffusion+DP 平均 original_acc", C.blue);
  addTable(s, n, 92, 374, [154, 198, 100, 100, 138, 138, 142], 42, [
    ["数据集", "最佳方法", "ε", "k", "分类器", "original_acc", "synthetic_acc"],
    ["MNIST", "Diffusion+DP", "0.5", "320", "ResNet20", "96.91%", "95.94%"],
    ["CIFAR10", "Diffusion+DP", "1.0", "320", "CNN", "47.66%", "46.88%"],
    ["结论", "Diffusion+DP 平均最高", "-", "-", "保持模型不变", "主指标", "辅助指标"],
  ], { size: 14 });
  addBulletList(s, n, [
    "MNIST 已接近可用，主要改进空间转为稳定性与重复实验。",
    "CIFAR10 明显更难，说明复杂彩色图像受合成质量和样本规模限制更大。",
  ], 112, 570, 1000, 38, { size: 18 });
  addFooter(s, n);
}

async function slide7(p) {
  const s = p.slides.add();
  const n = 7;
  addHeader(s, n, "只扩 Diffusion+DP：样本规模带来主要收益");
  addBarChart(s, n, 74, 198, 530, 340, "上一轮最优 vs 扩展后单次最优（%）", ["MNIST", "CIFAR10"], [
    { name: "完整矩阵", values: [96.91, 47.66], color: C.blue },
    { name: "扩展实验", values: [98.72, 61.40], color: C.green },
  ]);
  addLineChart(s, n, 646, 198, 530, 340, "CIFAR10 Diffusion+DP sweep: k 与 original_acc（%）", ["ε=0.7", "ε=1.0", "ε=1.5", "ε=2.0"], [
    { name: "k=640 CNN", values: [56.49, 57.65, 57.82, 57.13], color: C.blue },
    { name: "k=1000 CNN", values: [59.89, 61.40, 61.12, 59.88], color: C.green },
  ]);
  addBulletList(s, n, [
    "CIFAR10 从 47.66% 提升到 61.40%，主要来自 k=320 → 1000。",
    "MNIST 从 96.91% 提升到 98.72%，已进入稳定高精度区间。",
    "epsilon=1.0 与 1.5 在 CIFAR10 上差距很小，后续更该关注样本质量。",
  ], 106, 570, 1050, 34, { size: 17 });
  addFooter(s, n, "扩展口径：next_priority_experiments/diffusion_sweep_all.csv");
}

async function slide8(p) {
  const s = p.slides.add();
  const n = 8;
  addHeader(s, n, "最终结果：3 seeds 均值与标准差作为主结论");
  addBarChart(s, n, 70, 192, 590, 348, "最终候选 original_acc mean（%）", ["CIFAR ε=1.5 k=1000 CNN", "CIFAR ε=1.0 k=1000 CNN", "MNIST ε=0.5 k=640 ResNet20"], [
    { name: "mean", values: [60.32, 60.22, 98.54], color: C.green },
  ]);
  addTable(s, n, 700, 202, [180, 112, 154, 154], 52, [
    ["配置", "n", "original_acc", "训练时间"],
    ["CIFAR10 ε=1.5,k=1000,CNN", "3", "60.32%±1.44pp", "37.25s"],
    ["CIFAR10 ε=1.0,k=1000,CNN", "3", "60.22%±1.46pp", "37.40s"],
    ["MNIST ε=0.5,k=640,ResNet20", "3", "98.54%±0.14pp", "137.76s"],
  ], { size: 12 });
  addCard(s, n, 704, 466, 430, 86, "采用理由", "不直接采用单次最高值，而用 3 个随机种子均值和标准差降低随机训练波动对结论的影响。", C.blue);
  addBulletList(s, n, [
    "MNIST 波动很小，可作为阶段性稳定结果保留。",
    "CIFAR10 两个 ε 候选接近，说明当前瓶颈不在 1.0/1.5 的细小差异。",
  ], 106, 580, 1040, 36, { size: 17 });
  addFooter(s, n, "统计口径：next_priority_experiments/repeats_summary.csv");
}

async function slide9(p) {
  const s = p.slides.add();
  const n = 9;
  addHeader(s, n, "视觉对比：合成样本保留结构，但 CIFAR10 仍有质量瓶颈");
  addShape(s, n, "roundRect", 70, 190, 548, 330, C.light, C.line, 1, "image panel");
  addText(s, n, "CIFAR10: Diffusion+DP ε=1.5, k=1000", 94, 208, 410, 24, { size: 15, color: C.blue, bold: true, role: "image title" });
  await addImage(s, n, cifarImage, 94, 244, 500, 250, "contain", "cifar sample comparison");
  addShape(s, n, "roundRect", 664, 190, 548, 330, C.light, C.line, 1, "image panel");
  addText(s, n, "MNIST: Diffusion+DP ε=0.5, k=640", 688, 208, 410, 24, { size: 15, color: C.green, bold: true, role: "image title" });
  await addImage(s, n, mnistImage, 688, 244, 500, 250, "contain", "mnist sample comparison");
  addCard(s, n, 94, 548, 334, 82, "读图重点", "左侧为真实近邻，右侧为合成图；MNIST 结构更清晰，CIFAR10 保留语义但纹理仍弱。", C.blue);
  addCard(s, n, 474, 548, 334, 82, "质量指标", "修正后 SSIM/PSNR 去掉标题和白边影响，视觉指标应和 original_acc 联合解释。", C.green);
  addCard(s, n, 854, 548, 334, 82, "汇报口径", "合成图能否训练出真实域可用模型，最终仍以 original_acc 和重复实验稳定性判断。", C.amber);
  addFooter(s, n);
}

async function slide10(p) {
  const s = p.slides.add();
  const n = 10;
  addHeader(s, n, "结论与下一步");
  addMetric(s, n, 84, 184, 252, 130, "Diffusion+DP", "保留为主方法，GAN+DP 暂不继续扩展", C.green);
  addMetric(s, n, 364, 184, 252, 130, "original_acc", "作为核心排序指标，synthetic_acc 只辅助解释", C.blue);
  addMetric(s, n, 644, 184, 252, 130, "k 优先", "CIFAR10 提升主要来自扩大合成样本规模", C.amber);
  addMetric(s, n, 924, 184, 252, 130, "CIFAR10", "下一阶段主要攻关对象", C.red);
  addTable(s, n, 96, 372, [250, 742], 52, [
    ["下一步方向", "具体做法"],
    ["提高合成质量", "引入真实 DP-SGD U-Net 扩散模型、noise multiplicity、timestep 采样与 EMA。"],
    ["扩大有效样本", "CIFAR10 从 k=1000 继续到更大合成集，并增加低质量样本筛选。"],
    ["增强下游训练", "保持隐私口径一致，尝试预训练特征、ResNet/WRN、ensemble 与更充分训练。"],
    ["结果可信度", "最优候选继续做 3 seeds 以上重复，并报告均值、标准差和训练成本。"],
  ], { size: 14 });
  addFooter(s, n, "阶段性结论：MNIST 可作为成熟结果，CIFAR10 聚焦样本质量与训练策略");
}

async function build() {
  await ensureDirs();
  const p = Presentation.create({ slideSize: { width: W, height: H } });
  p.theme.colorScheme = {
    name: "Xidian DP Report",
    themeColors: {
      accent1: C.blue,
      accent2: C.green,
      accent3: C.amber,
      bg1: C.white,
      bg2: C.light,
      tx1: C.ink,
      tx2: C.muted,
    },
  };
  await slide1(p);
  await slide2(p);
  await slide3(p);
  await slide4(p);
  await slide5(p);
  await slide6(p);
  await slide7(p);
  await slide8(p);
  await slide9(p);
  await slide10(p);
  return p;
}

async function saveBlob(blob, filePath) {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  await fs.writeFile(filePath, bytes);
}

async function verifyAndExport(p) {
  inspect.unshift({ kind: "deck", slideCount: p.slides.count, slideSize: { width: W, height: H } });
  await fs.writeFile(INSPECT_PATH, inspect.map((r) => JSON.stringify(r)).join("\n") + "\n", "utf8");
  for (let i = 0; i < p.slides.items.length; i += 1) {
    const png = await p.export({ slide: p.slides.items[i], format: "png", scale: 1 });
    await saveBlob(png, path.join(PREVIEW_DIR, `slide-${String(i + 1).padStart(2, "0")}.png`));
  }
  const pptx = await PresentationFile.exportPptx(p);
  const out = path.join(OUT_DIR, "output.pptx");
  await pptx.save(out);
  console.log(out);
}

const presentation = await build();
await verifyAndExport(presentation);
