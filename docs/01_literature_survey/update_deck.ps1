$ErrorActionPreference = "Stop"
$path = "D:\sem5\Ds\cp\modi-lipi-xai\docs\01_literature_survey\Literature_Review_Presentation.pptx"
$pp = New-Object -ComObject PowerPoint.Application
$p = $pp.Presentations.Open($path, $false, $false, $false)

function SetText($slide, $starts, $new) {
  foreach ($sh in $p.Slides($slide).Shapes) {
    if ($sh.HasTextFrame -and $sh.TextFrame.HasText -and $sh.TextFrame.TextRange.Text.StartsWith($starts)) {
      $sh.TextFrame.TextRange.Text = $new
      return
    }
  }
  throw "Slide $slide has no text starting with: $starts"
}
function SetNotes($slide, $new) {
  $p.Slides($slide).NotesPage.Shapes.Placeholders(2).TextFrame.TextRange.Text = $new
}

try {
  # counts
  SetText 3 "7" "6"
  SetText 7 "Our 7 base papers" "Our 6 base papers"
  SetText 18 "Full details" "Full details of every paper are in our literature survey sheet (Literature_Survey_30_Papers.xlsx), with a second sheet for the 6 base papers."
  SetNotes 7 "We picked these 6 because each one feeds directly into one of our objectives."

  # paper 7 is no longer a base paper: remove its highlight in the table on slide 4
  $t = ($p.Slides(4).Shapes | Where-Object { $_.HasTable } | Select-Object -First 1).Table
  if ($t.Cell(8, 1).Shape.TextFrame.TextRange.Text -ne "7") { throw "row 8 of slide 4 is not paper 7" }
  1..5 | ForEach-Object {
    $c = $t.Cell(8, $_)
    $c.Shape.Fill.ForeColor.RGB = 16777215
    $c.Shape.TextFrame.TextRange.Font.Bold = 0
  }
  # paper 14: the paper never says raw pixels
  $t2 = ($p.Slides(5).Shapes | Where-Object { $_.HasTable } | Select-Object -First 1).Table
  if ($t2.Cell(5, 1).Shape.TextFrame.TextRange.Text -ne "14") { throw "row 5 of slide 5 is not paper 14" }
  $t2.Cell(5, 5).Shape.TextFrame.TextRange.Text = "No features described for SVM, KNN"

  # base paper slides (before deleting slide 10)
  SetText 8 "This benchmark size" "They say VGG16 helps most on letters that look alike, but give no numbers for it. We measure that pair by pair."
  SetNotes 8 "They used the public set of 46 letters. VGG16 overfits a bit, and the conclusion quotes a different number. They say it helps on similar letters but never measure it, and that's what objective 4 does."
  SetText 9 "Look-alike letters are the real problem" "A model should look harder at the strokes that separate similar letters. We do that with one small specialist per pair, not one attention layer for every letter."
  SetText 9 "Leads to" "Leads to Objectives 2 and 3"
  SetNotes 9 "This is the closest idea to ours. They add attention so the network focuses on the right strokes. We go further for the worst pairs: each pair gets its own small model that only looks at the regions that separate those two letters."
  SetText 11 "KNN and SVM were given raw pixels" "The paper doesn't describe any specific handcrafted features for KNN and SVM, so those scores say little about what an SVM can do."
  SetText 11 "Give the classical models" "We keep the SVM as our one model and give it proper features. 77% is the number to beat."
  SetText 11 "Leads to" "Leads to Objective 1"
  SetNotes 11 "They tried KNN, SVM and CNNs on the same images. The SVM got 77%, but they don't describe any handcrafted features for it. So we start from an SVM and give it proper features."
  SetText 12 "The features come from a deep network" "The features come from a deep network, so nobody can explain the decisions or point at a part of the letter."
  SetText 12 "We use Random Forest and XGBoost too" "We use handcrafted features instead, so SHAP can show which part of a letter separates a pair."
  SetNotes 12 "Good accuracy, but you can't point at a part of the letter with VGG16 features. Handcrafted features let SHAP do exactly that, and our specialists depend on it."
  SetText 13 "Otsu thresholding is the main" "Otsu thresholding is our main cleaning step, and look-alike letters are exactly where we focus."
  SetText 13 "Leads to" "Leads to Objectives 1 and 2"
  SetNotes 13 "An early system on very little data. Their useful finding is that similar shapes do worst, which is still true on our much bigger dataset."

  # the gap: one gap, not all of them
  SetText 16 "Four gaps we found" "The one gap we picked"
  SetText 16 "No proper features" "Everyone names the problem"
  SetText 16 "Classical models were given" "Paper after paper says look-alike letters are the hardest part."
  SetText 16 "Papers 14, 26, 28" "Papers 4, 6, 26"
  SetText 16 "Models never combined" "Nobody lists the pairs"
  SetText 16 "Random Forest and XGBoost were tried" "They talk about similar letters but don't say which ones get mixed up."
  SetText 16 "Paper 21" "Papers 6, 26"
  SetText 16 "No explanations" "Nobody shows what separates them"
  SetText 16 "Everyone says look-alike" "None of the papers shows which part of the letter tells two look-alikes apart."
  SetText 16 "Papers 6, 26" "Papers 4, 6, 21"
  SetText 16 "Big networks" "The usual fix is a bigger model"
  SetText 16 "CNNs trained on" "Most papers change the whole network instead of fixing those mistakes directly."
  SetText 16 "Papers 4, 7" "Papers 4, 6, 11"
  SetNotes 16 "There were other gaps too, but we picked just this one: look-alike letters. Everything in our project comes from it."

  # objectives
  SetText 17 "Recognise 57 handwritten" "Recognise 57 handwritten Modi characters with one model, an SVM on handcrafted features"
  SetText 17 "Based on Kulkarni" "Based on Mandke et al. 2024, Kulkarni et al. 2014, Anam & Gupta 2015"
  SetText 17 "Compare Random Forest" "Find which pairs of characters the SVM mixes up most, and use SHAP to see which parts separate them"
  SetText 17 "Based on Ekbote" "Based on Kolte & Gupta 2025, Anam & Gupta 2015, Ekbote et al. 2022"
  SetText 17 "Use SHAP to show" "Build a small specialist for each look-alike pair that looks only at those parts (our new idea)"
  SetText 17 "Based on Kolte" "Based on Kolte & Gupta 2025. None of the 30 papers does this."
  SetText 17 "Compare our model" "Check on a fixed test set whether the specialists really reduce mistakes, pair by pair"
  SetText 17 "Based on Bhong" "Based on Bhong & Kulkarni 2025"
  SetNotes 17 "These are the four objectives in our synopsis. One model, the SVM, and one new idea: when the SVM is stuck between two look-alike letters, a small specialist that only looks at the regions separating them makes the call."

  # paper 7 stays in the table; only its base-paper slide goes
  $p.Slides(10).Delete()
  $p.Save()
  "slides now: $($p.Slides.Count)"
  $out = "D:\sem5\Ds\cp\modi-lipi-xai\logs\deck_png"
  New-Item -ItemType Directory -Force $out | Out-Null
  foreach ($i in 4,5,9,10,15,16) { $p.Slides($i).Export("$out\s$i.png", "PNG", 960, 540) }
} finally {
  $p.Close()
  $pp.Quit()
}
