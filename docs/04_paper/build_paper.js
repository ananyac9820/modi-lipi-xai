// IEEE-style conference paper (two columns, US Letter) built with docx-js.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType, WidthType,
  BorderStyle, SectionType, TabStopType, VerticalAlign,
} = require("docx");

const HERE = __dirname;
const OUT = process.argv[2] || path.join(HERE, "Modi_Handcrafted_Features_IEEE.docx");
const FONT = "Times New Roman";
const COL_W = 5040; // 3.5 in column in DXA

// ---------------- references (numbered in order of first citation) ----------------
const REFS = {
  pandey: 'A. Pandey, "Proposal to encode the Modi script in ISO/IEC 10646," Unicode Technical Committee, Doc. L2/11-212R2, 2011.',
  bhong: 'M. M. Bhong and N. Kulkarni, "Handwritten Modi script character recognition using CNN and VGG16 algorithms," Int. J. Adv. Res. Sci. Commun. Technol., 2025, doi: 10.48175/IJARSCT-25317.',
  kolte: 'S. Kolte and B. Gupta, "Modi script character recognition using convolutional neural networks and attention-based CNN architecture," Int. J. Adv. Res. Sci. Commun. Technol., 2025, doi: 10.48175/IJARSCT-25309.',
  chandankhede: 'C. Chandankhede, R. Sachdeo, U. Shrawankar, N. Wankhade, and G. Talmale, "Modi script handwritten character classification using drop-out induced incremental learning approach," J. Inf. Syst. Eng. Manage., vol. 10, no. 10s, 2025, doi: 10.52783/jisem.v10i10s.1403.',
  mandke: 'S. Mandke, V. Pimprale, and P. Date, "Offline MODI script character recognition using deep learning techniques," Library Progress International, 2024.',
  sonavane: 'R. Sonavane et al., "Exploring ResNet101, InceptionV3, and Xception for Modi script character classification," Int. J. Intell. Syst. Appl. Eng., 2024.',
  ekbote: 'M. Ekbote, A. Jadhav, and D. D. Ambawade, "Implementing a hybrid deep learning approach to achieve classic handwritten alphanumeric MODI recognition," Int. J. Eng. Adv. Technol., 2022, doi: 10.35940/ijeat.A3846.1012122.',
  anam: 'S. Anam and S. Gupta, "An approach for recognizing Modi Lipi using Ostu\'s binarization algorithm and Kohenen neural network," Int. J. Comput. Appl., 2015, doi: 10.5120/19511-1128.',
  kulkarni: 'S. A. Kulkarni, P. L. Borde, R. R. Manza, and P. L. Yannawar, "Recognition of handwritten MODI numerals using Hu and Zernike features," arXiv:1404.1151, 2014.',
  kausadikar: 'H. Kausadikar, T. Kale, O. Susladkar, and S. Mittal, "Historic scripts to modern vision: A novel dataset and a VLM framework for transliteration of Modi script to Devanagari," arXiv:2503.13060, 2025.',
  joseph: 'S. Joseph and J. P. George, "Machine transliteration of handwritten MODI script to Devanagari using deep neural networks," Int. J. Comput., vol. 23, no. 4, 2024, doi: 10.47839/ijc.23.4.3540.',
  varpe: 'K. Varpe and S. R. Sakhare, "Review of character recognition techniques for MODI script," Indian J. Sci. Technol., vol. 16, no. 26, 2023, doi: 10.17485/IJST/v16i26.485.',
  dataset: '"MODI-HDC: Historical handwritten Modi script dataset" (MODI-HChar character subset), Kaggle. [Online]. Available: https://www.kaggle.com/',
  otsu: 'N. Otsu, "A threshold selection method from gray-level histograms," IEEE Trans. Syst., Man, Cybern., vol. SMC-9, no. 1, pp. 62-66, 1979.',
  dalal: 'N. Dalal and B. Triggs, "Histograms of oriented gradients for human detection," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2005, pp. 886-893.',
  khotanzad: 'A. Khotanzad and Y. H. Hong, "Invariant image recognition by Zernike moments," IEEE Trans. Pattern Anal. Mach. Intell., vol. 12, no. 5, pp. 489-497, 1990.',
  ojala: 'T. Ojala, M. Pietikainen, and T. Maenpaa, "Multiresolution gray-scale and rotation invariant texture classification with local binary patterns," IEEE Trans. Pattern Anal. Mach. Intell., vol. 24, no. 7, pp. 971-987, 2002.',
  breiman: 'L. Breiman, "Random forests," Mach. Learn., vol. 45, no. 1, pp. 5-32, 2001.',
  cortes: 'C. Cortes and V. Vapnik, "Support-vector networks," Mach. Learn., vol. 20, no. 3, pp. 273-297, 1995.',
  xgb: 'T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Min., 2016, pp. 785-794.',
  wolpert: 'D. H. Wolpert, "Stacked generalization," Neural Netw., vol. 5, no. 2, pp. 241-259, 1992.',
  shap: 'S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2017, pp. 4765-4774.',
  sklearn: 'F. Pedregosa et al., "Scikit-learn: Machine learning in Python," J. Mach. Learn. Res., vol. 12, pp. 2825-2830, 2011.',
};
const order = [];
const cite = (s) => s.replace(/\[\[([a-z,]+)\]\]/g, (_, keys) => {
  const nums = keys.split(",").map((k) => {
    if (!REFS[k]) throw new Error("unknown ref " + k);
    if (!order.includes(k)) order.push(k);
    return order.indexOf(k) + 1;
  });
  return nums.map((n) => `[${n}]`).join(", ");
});

// ---------------- inline markup: **bold**, *italic*, ^sup^, _sub_ ----------------
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|\^[^^]+\^|_\{[^}]+\})/g;
  let last = 0, m;
  text = cite(text);
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, font: FONT, ...base }));
    else if (t.startsWith("*")) out.push(new TextRun({ text: t.slice(1, -1), italics: true, font: FONT, ...base }));
    else if (t.startsWith("^")) out.push(new TextRun({ text: t.slice(1, -1), superScript: true, font: FONT, ...base }));
    else out.push(new TextRun({ text: t.slice(2, -1), subScript: true, font: FONT, ...base }));
    last = re.lastIndex;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, ...base }));
  return out;
}

const P = (text, opts = {}) => new Paragraph({
  children: runs(text, { size: 20 }), alignment: AlignmentType.JUSTIFIED,
  indent: { firstLine: opts.noIndent ? 0 : 202 }, spacing: { after: 0, line: 228 }, ...opts.p,
});
let h1n = 0, h2n = 0;
const ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"];
const H1 = (t) => { h2n = 0; return new Paragraph({ children: [new TextRun({ text: `${ROMAN[h1n++]}. ${t}`, font: FONT, size: 20, smallCaps: true })], alignment: AlignmentType.CENTER, spacing: { before: 200, after: 100 }, keepNext: true }); };
const H2 = (t) => new Paragraph({ children: [new TextRun({ text: `${String.fromCharCode(65 + h2n++)}. ${t}`, font: FONT, size: 20, italics: true })], spacing: { before: 100, after: 60 }, keepNext: true });
const EQ = (left, num) => new Paragraph({
  children: [...runs(left, { size: 20, italics: false }), new TextRun({ text: `\t(${num})`, font: FONT, size: 20 })],
  tabStops: [{ type: TabStopType.CENTER, position: 2520 }, { type: TabStopType.RIGHT, position: COL_W }],
  spacing: { before: 60, after: 60 }, indent: { left: 0 },
});
const LIST = (items) => items.map((t, i) => new Paragraph({
  children: runs(`${i + 1}) ${t}`, { size: 20 }), alignment: AlignmentType.JUSTIFIED,
  indent: { left: 202, hanging: 0 }, spacing: { after: 20, line: 228 },
}));
let figN = 0, tabN = 0;
function FIG(file, wIn, caption) {
  const buf = fs.readFileSync(path.join(HERE, file));
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
  const wPx = Math.round(wIn * 96), hPx = Math.round(wPx * h / w);
  figN++;
  return [
    new Paragraph({ children: [new ImageRun({ type: "png", data: buf, transformation: { width: wPx, height: hPx }, altText: { title: caption, description: caption, name: file } })], alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 }, keepNext: true }),
    new Paragraph({ children: runs(`Fig. ${figN}. ${caption}`, { size: 16 }), alignment: AlignmentType.JUSTIFIED, spacing: { after: 140 } }),
  ];
}
function TABLE(title, header, rows, widths) {
  tabN++;
  const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  const line = { style: BorderStyle.SINGLE, size: 6, color: "000000" };
  const mk = (cells, ri, last) => new TableRow({
    cantSplit: true,
    children: cells.map((c, ci) => new TableCell({
      width: { size: widths[ci], type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER,
      margins: { top: 20, bottom: 20, left: 50, right: 50 },
      borders: { top: ri === 0 ? line : none, bottom: ri === 0 || last ? line : none, left: none, right: none },
      children: [new Paragraph({ children: runs(String(c), { size: 16, bold: ri === 0 }), alignment: ci === 0 ? AlignmentType.LEFT : AlignmentType.CENTER, keepNext: !last, keepLines: true })],
    })),
  });
  const R = ["I", "II", "III", "IV"];
  return [
    new Paragraph({ children: [new TextRun({ text: `TABLE ${R[tabN - 1]}`, font: FONT, size: 16 })], alignment: AlignmentType.CENTER, spacing: { before: 140, after: 0 }, keepNext: true }),
    new Paragraph({ children: [new TextRun({ text: title, font: FONT, size: 16, smallCaps: true })], alignment: AlignmentType.CENTER, spacing: { after: 60 }, keepNext: true }),
    new Table({ columnWidths: widths, width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA }, alignment: AlignmentType.CENTER,
      rows: [mk(header, 0, false), ...rows.map((r, i) => mk(r, i + 1, i === rows.length - 1))] }),
    new Paragraph({ children: [], spacing: { after: 100 } }),
  ];
}

// ---------------- content ----------------
const body = [];
const add = (...xs) => xs.flat().forEach((x) => body.push(x));

// Abstract and index terms
add(new Paragraph({
  children: [new TextRun({ text: "Abstract", font: FONT, size: 18, bold: true, italics: true }), new TextRun({ text: "—", font: FONT, size: 18, bold: true }),
    ...runs("Modi is a cursive script that was used to write Marathi for several centuries, and many land, court and family records in Maharashtra's archives are written in it. Papers on recognising handwritten Modi agree that characters which look alike are the hardest part, but they rarely say which characters these are or what separates them, and the usual response is a bigger network. We tried something smaller. One SVM on handcrafted features (HOG, Zernike moments and LBP, reduced with LDA) reads every character. When its top two guesses are a known look-alike pair, a small binary SVM for that pair makes the final call, and this specialist sees only the regions of the character that SHAP ranks highest for that pair. The pairs, the regions and the decision to keep each specialist all come from training data. On the public MODI-HChar dataset (57 classes) with a fixed test set of 11,400 images, the SVM alone reaches 86.36% and the SVM with 12 specialists reaches 86.78%. The gain is small but statistically significant (182 images fixed, 134 broken, McNemar p = 0.008). Results differ by pair: nine pairs improve, including i/ii, whose error rate falls from 23% to 18%, and three get worse.", { size: 18, bold: true })],
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 100, line: 216 },
}));
add(new Paragraph({
  children: [new TextRun({ text: "Index Terms", font: FONT, size: 18, bold: true, italics: true }), new TextRun({ text: "—", font: FONT, size: 18, bold: true }),
    ...runs("Modi script, handwritten character recognition, handcrafted features, support vector machine, SHAP, similar characters.", { size: 18, bold: true })],
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 120, line: 216 },
}));

add(H1("Introduction"));
add(P("Modi is a cursive form of Nagari that was used for everyday Marathi writing, especially for revenue records, court papers and letters, until Devanagari replaced it in the twentieth century [[pandey]]. Much of this material survives in archives, but very few people can still read the script, so most of it has never been transcribed. Character recognition would not solve that on its own, but any reading system needs it."));
add(P("Writers of Modi kept the pen on the paper, so letters hang from a shared top line. Several letters differ only by a small hook or loop. Paper after paper names this as the main difficulty [[kolte,anam]], and a model's overall accuracy hides it. On our data, for example, an SVM with 86% accuracy overall mixes up i and ii in 23% of their test images."));
add(P("We wanted to deal with these mistakes directly and leave the rest of the model alone. Our idea is a small specialist for each look-alike pair. It looks only at the parts of the character that tell those two apart, and it is asked only when the main model is stuck between them. SHAP [[shap]] tells us which parts to use. This is why we use handcrafted features: a HOG or LBP value belongs to a known region of the image, so a SHAP ranking can be turned directly into a choice of inputs."));
add(P("The paper has four aims:"));
add(LIST([
  "recognise 57 Modi characters with one model, an SVM on handcrafted features;",
  "find the pairs it mixes up most, and use SHAP to see which regions separate each pair;",
  "build a specialist for each pair that sees only those regions, and let it decide when the SVM's top two guesses are that pair;",
  "check on a fixed test set whether the specialists reduce mistakes, reporting each pair separately, including the pairs that get worse.",
]));

add(H1("Related Work"));
add(P("Most recent work trains convolutional networks. Bhong and Kulkarni [[bhong]] compared a small CNN with a fine-tuned VGG16 on a public set of 46 classes with 90 images each and reported 89.5% validation accuracy for VGG16. They say it does better on similar characters but give no numbers for them. Kolte and Gupta [[kolte]] added spatial and channel attention to a CNN to separate similar characters and reported 94.6%. This is the closest idea to ours, but one attention mechanism serves every class, and the paper does not list which characters were confused. Chandankhede et al. [[chandankhede]] and Sonavane et al. [[sonavane]] tried deeper networks and transfer learning."));
add(P("Classical models get less attention. Mandke et al. [[mandke]] ran KNN, SVM and several CNNs on the same images. The SVM scored 77% against 96% for the best CNN, but the paper does not describe any specific handcrafted features for it. Ekbote et al. [[ekbote]] classified VGG16 features with Random Forest and XGBoost and reached 92% on characters, although learned features like these cannot be pointed at on the character. Anam and Gupta [[anam]] used Otsu binarisation with a Kohonen network on 22 characters and noted that similar shapes did worst. Kulkarni et al. [[kulkarni]] found that Zernike moments beat Hu moments on 100 Modi digit images (86.66% against 70%)."));
add(P("Other groups work on whole lines or documents [[joseph,kausadikar]], and Varpe and Sakhare [[varpe]] review the field. Published accuracies use different datasets, class counts and splits, so we do not rank our result against them. None of the 30 Modi papers we surveyed builds a separate model for a confused pair or uses SHAP to choose its inputs. Second-stage classifiers for confusing characters have been tried on other scripts, so we claim this as new for Modi, not as a new idea in character recognition generally."));

add(H1("Dataset and Experimental Setup"));
add(P("We use MODI-HChar, the isolated-character part of the MODI-HDC collection on Kaggle [[dataset]]. After dropping unreadable files it has 575,912 images in 57 classes: 35 consonants, 12 vowels and 10 digits, with about 10,100 images per class. We counted these from disk."));
add(P("The images of every class are shuffled once with a fixed seed. The first 200 images of each class form the test set (11,400 images), and the next 500 form the training set (28,500 images). Every decision about the specialists uses 3-fold out-of-fold predictions on the training set only (Fig. 1). Everything runs on a laptop CPU."));
add(FIG("figures/protocol.png", 2.7, "Evaluation protocol. The pairs, the regions and the choice of which specialists to keep all come from training images. The test set is scored at the end."));

add(H1("Method"));
add(P("Fig. 2 shows the full pipeline. The implementation uses scikit-learn [[sklearn]], scikit-image, mahotas and SHAP."));
add(FIG("figures/pipeline.png", 2.45, "The recognition pipeline. The numbers in brackets are the lengths of the feature vectors."));
add(H2("Preprocessing"));
add(P("Each image is converted to grayscale, median filtered and binarised with Otsu's method [[otsu]]. Scans are not consistent about polarity, so we treat the minority class of pixels as ink. Frame lines along the border, long thin vertical grid lines and small specks are removed. The character is then cropped to its ink and scaled, keeping its aspect ratio, into a 64 x 64 image."));
add(H2("Handcrafted Features"));
add(P("*HOG* [[dalal]] describes stroke direction. With 8 x 8 cells, 2 x 2 blocks and 9 orientation bins, a 64 x 64 image gives 1,764 values. *Zernike moments* [[khotanzad]] describe the overall shape and do not change when the character rotates; magnitudes up to order 12 give 49 values. *LBP* [[ojala]] captures small patterns such as stroke ends and junctions. We use uniform patterns at two scales (8 points at radius 1, 16 points at radius 2) pooled over a 4 x 4 grid, which gives 448 values and keeps a rough idea of where each pattern occurs."));
add(H2("The Main SVM"));
add(P("Features are standardised on the training set. HOG has 36 times more values than Zernike, so each family *f* with *d*_{f} values is multiplied by"));
add(EQ("\tw_{f} = 1 / √d_{f}", 1));
add(P("and the 2,261 values are reduced with shrinkage LDA to at most 56 dimensions. We chose LDA over PCA on training data, where 3-fold cross-validation gave 84.5% against 79.7%. The classifier is an RBF SVM [[cortes]] with *C* = 10. We picked it on the same training folds, where it scored 85.5% against 83.4% for Random Forest [[breiman]]. On the test set, Random Forest (84.0%), XGBoost [[xgb]] (83.8%) and a stack of all three [[wolpert]] (84.8%) also came in below the SVM (86.4%), so we dropped them and kept one model.", { noIndent: true }));
add(H2("Finding the Look-alike Pairs"));
add(P("We split the training set into 3 folds and collect the SVM's out-of-fold guesses. Counting its errors gives the 15 most confused pairs. For each pair we also take the top two classes by SVM decision score, which tells us when the specialist would be asked."));
add(H2("Choosing Regions with SHAP"));
add(P("Explaining 2,261 separate numbers would not help anyone, so we group them into 35 groups a person can point to: HOG pooled into 16 regions on a 4 x 4 grid, LBP into the same 16 regions, and Zernike moments into three bands. For each pair we train a binary SVM on that pair's training images and run Kernel SHAP [[shap]] on 30 of them, with 150 sampled coalitions per image and the mean feature vector as the reference. Contributions add up to the model output:"));
add(EQ("\tf(x) = φ_{0} + Σ_{g} φ_{g}", 2));
add(P("where φ_{g} is the contribution of group *g*. The 10 groups with the largest mean |φ_{g}| are that pair's regions.", { noIndent: true }));
add(H2("The Specialists"));
add(P("A specialist is an RBF SVM (*C* = 1, balanced class weights) trained on one pair's images using only the standardised columns of its 10 regions. It skips LDA, so it works on the regions directly. If the two highest SVM scores for an image *x* belong to classes *a* and *b*, the prediction is"));
add(EQ("\tŷ = s_{ab}(x) if {a, b} ∈ P, otherwise ŷ = a", 3));
add(P("where *a* is the top class, *s*_{ab} is the specialist and *P* is the set of kept pairs. We keep a specialist only if it gets more images right than the SVM does on held-out training images. Twelve of the 15 passed. Over the 15 candidates, the SVM was right on 6,830 of the 7,900 held-out images that would go to a specialist; with the 15 specialists it was 6,917.", { noIndent: true }));

add(H1("Results"));
add(H2("Overall Accuracy"));
add(P("Table I gives the result on the 11,400 test images. The specialists raise accuracy from 86.36% to 86.78%. They fixed 182 images the SVM got wrong and broke 134 it got right, and an exact McNemar test on these paired predictions gives p = 0.008. (The SVM's ordinary prediction is a one-vs-one vote. Ranking by decision score, which is how we find the top two, gives 86.27%, and against that number the specialists fixed 161 and broke 103, p = 0.0004.) Half a point is not much on paper, but the specialists only act on 26% of the images, and on those 2,936 images correct answers go from 2,402 (81.8%) to 2,460 (83.8%). All three character types improve, digits and vowels more than consonants."));
add(TABLE("Test accuracy (11,400 images, 500 training images per class)",
  ["Model", "Acc. (%)", "Macro-F1", "Digits", "Vowels", "Cons."],
  [["SVM alone", "86.36", "0.863", "92.7", "88.9", "83.7"],
   ["**SVM + specialists**", "**86.78**", "**0.868**", "**93.3**", "**89.8**", "**83.9**"]],
  [1500, 740, 760, 680, 680, 680]));
add(H2("Pair by Pair"));
add(P("Table II and Fig. 3 show each specialist on the images it decided. Nine of the 12 improve. The biggest change is i/ii, where the mix-up rate over the 400 test images of the pair falls from 23% to 18%. The main region SHAP picked for it is the top right of the character, which is where the extra hook of ii sits, so the choice also makes sense to a person looking at the letter."));
add(P("Three pairs get worse even though they passed the training check: ja/n loses 8 images, h/thha loses 5 and r/th loses 1. Passing that check is not a guarantee, since these gains on training folds can be small and noisy. h/thha is also the most confused pair in the dataset, and in Modi these two letters are drawn almost the same way, so a few local regions may simply not contain the answer. After the specialists, the most confused pairs are h/thha (79 errors), i/ii (72), p/thha (58), ja/n (53) and d/th (41)."));
const LOOK = JSON.parse(fs.readFileSync(path.join(HERE, "../../results/metrics/lookalike.json"), "utf8"));
const shortRegion = (r) => r.replace("stroke direction, ", "").replace("local stroke pattern, ", "").replace(": ", ", ");
add(TABLE("Each specialist on the test images it decided",
  ["Pair", "Images", "SVM right", "Final right", "Change", "Main region"],
  [...LOOK.per_pair].sort((a, b) => (b.correct_final - b.correct_svm) - (a.correct_final - a.correct_svm)).map((p) => {
    const d = p.correct_final - p.correct_svm;
    return [p.pair.join(" / "), p.images_consulted, p.correct_svm, p.correct_final, (d > 0 ? "+" : "") + d, shortRegion(p.regions[0])];
  }),
  [860, 560, 600, 620, 560, 1840]));
add(FIG("figures/fig_pairs.png", 3.2, "Wrong answers on the images each specialist decided, before and after. The number beside each pair is the change in correct answers."));
add(H2("Does SHAP Really Help?"));
add(P("We checked this in an early run of the same experiment. Specialists on the SHAP regions gave 86.71% (146 fixed, 96 broken, p = 0.002 against the SVM's score ranking). Specialists on all 2,261 features, with everything else the same, gave 86.46% (97 fixed, 75 broken, p = 0.11), which is not a significant gain. So narrowing each specialist to its own regions is what makes it work. The numbers in Table I come from a later rerun with the SHAP random numbers seeded, which is why 86.71% became 86.78%."));
add(H2("What Each Feature Adds"));
add(P("Fig. 4 shows the main SVM with different features. HOG alone reaches 83.6%, LBP alone 79.3% and Zernike alone 54.0%. Taking one out of the full set is more telling: without HOG accuracy falls to 81.4%, without LBP to 83.9%, and without Zernike only to 86.0%. So Zernike moments, which earlier work recommended for Modi digits [[kulkarni]], add little once HOG and LBP are in."));
add(FIG("figures/fig_ablation.png", 3.2, "SVM accuracy with single descriptors (light), pairs that leave one descriptor out (grey) and all three (dark)."));
add(H2("What SHAP Says about the SVM"));
add(P("We also ran the same grouped SHAP on the main SVM for 171 test images, three per class. In Fig. 5(a), HOG is 78% of the inputs and gets 66% of the evidence, LBP is 20% and gets 31%, and Zernike is 2% and gets about 3.5%. In Fig. 5(b), the four central HOG cells hold about 52% of the HOG evidence and the top row holds 15%. Our guess is that the headline runs across the top of nearly every character and so says little about which one it is. Look-alike pairs are different: for i/ii the most useful regions are at the top. To check the explanations are faithful, we replaced the three highest-ranked groups of each image with their reference values. The probability of the predicted class fell by 0.46 on average, against 0.05 for three random groups."));
add(FIG("figures/fig_shap.png", 3.2, "(a) Share of input features and share of SHAP evidence for each descriptor family. (b) Mean absolute SHAP value of the 16 HOG regions on a 4 x 4 grid over the character."));

add(H1("Discussion and Limitations"));
add(P("The overall gain is small. A deeper model or more training data would probably change accuracy by more than our specialists do, and adding them costs 12 extra models and some explaining. What they give in return is a fix aimed at a named mistake. We can say which pair improved, by how much and which part of the letter the specialist looked at, and we can see when it did not work, as with ja/n."));
add(P("There are limits to how far these numbers can be pushed. We looked at the test set twice, once in the early run and once in the seeded rerun, and we changed no settings between them, but the choice to go ahead with SHAP regions was made after seeing the first result. The p-values come from one fixed test set, and we made no correction for looking at 12 pairs separately, so the smaller per-pair changes in Table II may be noise. The specialist settings (15 pairs, 10 regions, *C* = 1) were chosen by us without a search."));
add(P("The system reads isolated characters. Our app can split a line of separated characters, but on synthetic words whose characters touch along the headline the segmenter gets the character count right only 42% of the time, so real cursive pages are still out of reach. The dataset also comes from a limited group of writers, and the Devanagari names we show for each class have not yet been checked by a Modi reader."));

add(H1("Conclusion"));
add(P("We recognised 57 handwritten Modi characters with one SVM on handcrafted features and added one new step: a small specialist for each look-alike pair, which sees only the regions SHAP picks for that pair. On a fixed test set of 11,400 images, accuracy went from 86.36% to 86.78%, nine of the 12 pairs improved and three got worse. Using all the features instead of the SHAP regions gave no significant gain. Next we would like to test this on a second dataset and find out why pairs like h/thha resist it."));

add(new Paragraph({ children: [new TextRun({ text: "Acknowledgment", font: FONT, size: 20, smallCaps: true })], alignment: AlignmentType.CENTER, spacing: { before: 200, after: 100 } }));
add(P("We thank our guide, Prof. Nilam Honmane, Department of Computer Engineering, Vishwakarma Institute of Technology, Pune, for guidance throughout this project."));

add(new Paragraph({ children: [new TextRun({ text: "References", font: FONT, size: 20, smallCaps: true })], alignment: AlignmentType.CENTER, spacing: { before: 200, after: 100 } }));
// references are added after all citations are resolved
const refParas = () => order.map((k, i) => new Paragraph({
  children: [new TextRun({ text: `[${i + 1}]`, font: FONT, size: 16 }), new TextRun({ text: `\t${REFS[k]}`, font: FONT, size: 16 })],
  tabStops: [{ type: TabStopType.LEFT, position: 360 }], indent: { left: 360, hanging: 360 },
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 30, line: 200 },
}));
const unused = Object.keys(REFS).filter((k) => !order.includes(k));
if (unused.length) throw new Error("uncited refs: " + unused.join(", "));

// ---------------- title block ----------------
const authorCell = (name, mail) => new TableCell({
  width: { size: 5040, type: WidthType.DXA },
  borders: { top: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" }, bottom: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" }, left: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" }, right: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" } },
  children: [
    new Paragraph({ children: [new TextRun({ text: name, font: FONT, size: 22 })], alignment: AlignmentType.CENTER }),
    new Paragraph({ children: [new TextRun({ text: "Department of Computer Engineering", font: FONT, size: 20, italics: true })], alignment: AlignmentType.CENTER }),
    new Paragraph({ children: [new TextRun({ text: "Vishwakarma Institute of Technology", font: FONT, size: 20, italics: true })], alignment: AlignmentType.CENTER }),
    new Paragraph({ children: [new TextRun({ text: "Pune, India", font: FONT, size: 20 })], alignment: AlignmentType.CENTER }),
    new Paragraph({ children: [new TextRun({ text: mail, font: FONT, size: 20 })], alignment: AlignmentType.CENTER }),
  ],
});
const title = [
  new Paragraph({ children: [new TextRun({ text: "Handwritten Modi Character Recognition using Handcrafted Features", font: FONT, size: 48 })], alignment: AlignmentType.CENTER, spacing: { after: 240 } }),
  new Table({ columnWidths: [5040, 5040], width: { size: 10080, type: WidthType.DXA }, alignment: AlignmentType.CENTER,
    rows: [new TableRow({ children: [authorCell("Ananya Viraj Choudhari", "viraj.ananya24@vit.edu"), authorCell("Arya Bharat Patil", "arya.patil24@vit.edu")] })] }),
  new Paragraph({ children: [], spacing: { after: 160 } }),
];

const page = { size: { width: 12240, height: 15840 }, margin: { top: 1080, bottom: 1440, left: 1080, right: 1080 } };
const doc = new Document({
  creator: "Ananya Viraj Choudhari, Arya Bharat Patil",
  title: "Handwritten Modi Character Recognition using Handcrafted Features",
  styles: { default: { document: { run: { font: FONT, size: 20 } } } },
  sections: [
    { properties: { page }, children: title },
    { properties: { page, type: SectionType.CONTINUOUS, column: { count: 2, space: 360, equalWidth: true } }, children: [...body, ...refParas()] },
  ],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(OUT, b); console.log("wrote", OUT, "refs", order.length, "figs", figN, "tables", tabN); });
