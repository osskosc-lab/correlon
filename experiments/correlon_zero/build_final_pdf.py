"""Build the final Japanese Correlon Zero submission PDF from frozen artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
OUTPUT = ROOT / "output" / "pdf" / "CORRELON_ZERO_FINAL_REPORT.pdf"
FONT = "HeiseiKakuGo-W5"
INK = colors.HexColor("#172033")
BLUE = colors.HexColor("#2563eb")
ORANGE = colors.HexColor("#ea580c")
PALE = colors.HexColor("#edf3fb")


def fmt(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def make_styles() -> dict[str, ParagraphStyle]:
    pdfmetrics.registerFont(UnicodeCIDFont(FONT))
    sample = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title", parent=sample["Title"], fontName=FONT, fontSize=25, leading=33,
            textColor=INK, alignment=TA_CENTER, spaceAfter=14,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=sample["Normal"], fontName=FONT, fontSize=12, leading=18,
            textColor=colors.HexColor("#596579"), alignment=TA_CENTER, spaceAfter=20,
        ),
        "heading": ParagraphStyle(
            "heading", parent=sample["Heading2"], fontName=FONT, fontSize=15, leading=21,
            textColor=INK, spaceBefore=10, spaceAfter=7,
        ),
        "body": ParagraphStyle(
            "body", parent=sample["BodyText"], fontName=FONT, fontSize=9.4, leading=15,
            textColor=INK, spaceAfter=7,
        ),
        "small": ParagraphStyle(
            "small", parent=sample["BodyText"], fontName=FONT, fontSize=8.0, leading=11,
            textColor=INK,
        ),
        "caption": ParagraphStyle(
            "caption", parent=sample["BodyText"], fontName=FONT, fontSize=7.8, leading=10.5,
            textColor=colors.HexColor("#596579"), alignment=TA_CENTER, spaceAfter=8,
        ),
        "verdict": ParagraphStyle(
            "verdict", parent=sample["Heading1"], fontName=FONT, fontSize=18, leading=25,
            textColor=colors.HexColor("#b42318"), alignment=TA_CENTER, spaceAfter=14,
        ),
    }


def paragraph(text: str, styles: dict[str, ParagraphStyle], name: str = "body") -> Paragraph:
    return Paragraph(text, styles[name])


def table(rows: list[list[str]], widths: list[float]) -> Table:
    rendered = [[Paragraph(str(cell), ParagraphStyle("cell", fontName=FONT, fontSize=7.5, leading=9.4)) for cell in row] for row in rows]
    output = Table(rendered, colWidths=widths, repeatRows=1, hAlign="LEFT")
    output.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), FONT),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return output


def figure(filename: str, caption: str, styles: dict[str, ParagraphStyle], width: float = 16.7 * cm) -> KeepTogether:
    image_path = FIGURES / filename
    image = Image(str(image_path))
    ratio = image.imageHeight / image.imageWidth
    image.drawWidth = width
    image.drawHeight = width * ratio
    return KeepTogether([image, Spacer(1, 0.12 * cm), paragraph(caption, styles, "caption")])


def footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#d8dee9"))
    canvas.line(1.7 * cm, 1.35 * cm, A4[0] - 1.7 * cm, 1.35 * cm)
    canvas.setFont(FONT, 7.5)
    canvas.setFillColor(colors.HexColor("#596579"))
    canvas.drawString(1.7 * cm, 0.85 * cm, "Correlon Zero - Adversarial Invariance Falsification v1.0")
    canvas.drawRightString(A4[0] - 1.7 * cm, 0.85 * cm, f"Page {document.page}")
    canvas.restoreState()


def build() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "correlon_zero_summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)

    styles = make_styles()
    corr = summary["metric_summary"]["correlon_v1"]
    baseline = summary["baseline_comparison"]
    audit = summary["adversarial_audit"]
    representation = summary["representation_audit"]
    fpr = [row for row in summary["false_positive_audit"] if row["metric"] == "correlon_v1"]
    fpr.sort(key=lambda row: row["false_positive_rate"], reverse=True)
    common = next(row for row in fpr if row["negative_class"] == "common_driver")
    low_rank = next(row for row in fpr if row["negative_class"] == "matched_low_rank")

    story = [
        Spacer(1, 2.0 * cm),
        paragraph("Correlon Zero", styles, "title"),
        paragraph("敵対的不変性反証実験 最終提出レポート", styles, "subtitle"),
        paragraph("CORRELON_ZERO_ADVERSARIAL_INVARIANCE_FALSIFICATION_v1.0", styles, "subtitle"),
        paragraph(summary["decision"]["primary_label"], styles, "verdict"),
        paragraph(
            "結論: 凍結した CorrelonZero_v1 は、対象の関係機構を一般的な相関、低ランク、共通原因、予測依存、"
            "持続性から分離する操作的定義として生き残らなかった。これは理論の失敗ではなく、再現可能な操作的反証の成功である。",
            styles,
        ),
        Spacer(1, 0.4 * cm),
        paragraph("提出情報", styles, "heading"),
        table([
            ["項目", "値"],
            ["確認的seed", "10000..10199 (200 seed)、各seedで3 targetを保守的に集約"],
            ["敵対探索", "300 admissible null trials、direct X-Y edge は常に absent"],
            ["科学的freeze", "commit cb07d84、implementation SHA-256 " + summary["scientific_freeze_manifest"]["combined_sha256"]],
            ["最終成果物commit", "f88369af74dcb773027165762f7c56eed22dc0af"],
            ["検証", "変換unit test 10/10 passed、独立結果検証の全10 check passed"],
        ], [4.2 * cm, 12.5 * cm]),
        PageBreak(),
        paragraph("1. 凍結仮説と操作的定義", styles, "heading"),
        paragraph(
            "候補は P1-P7 の保存変換で安定し、D1-D8 の破壊変換で消失しなければならない。"
            "各世界で P=min(P1..P7)、D=max(D1..D8)、Delta=P-D とし、各seedでは3 targetの最悪値を用いる。",
            styles,
        ),
        table([
            ["要素", "凍結仕様"],
            ["演算子", "windowed whitened cross-covariance SVD: Q=Cxx^(-1/2) Cxy Cyy^(-1/2)"],
            ["主スカラー", "CorrelonZero_v1=sqrt(clip(T_iso,0,1)*clip(T_floor,0,1))"],
            ["T_iso", "6個のcircular-shift nullに対するdominant singular-gap excess"],
            ["T_floor", "同nullに対する10th percentile adjacent-mode continuity excess"],
            ["window", "140; step=40; max_lag=3; epsilon=1e-5"],
            ["陽性閾値", fmt(summary["positive_thresholds"]["correlon_v1"], 8)],
            ["判定閾値", "mean Delta >= 0.10; lower-tail Delta > 0; FPR <= 0.05; baseline advantage >= 0.10"],
        ], [4.2 * cm, 12.5 * cm]),
        paragraph("2. 主結果", styles, "heading"),
        table([
            ["指標", "観測値"],
            ["mean P", fmt(corr["P"]["mean"], 6)],
            ["mean D", fmt(corr["D"]["mean"], 6)],
            ["mean Delta", fmt(corr["Delta"]["mean"], 6)],
            ["Delta 95% CI", f"[{fmt(corr['Delta']['mean_CI95'][0], 6)}, {fmt(corr['Delta']['mean_CI95'][1], 6)}]"],
            ["Delta 5th percentile", fmt(corr["Delta"]["q05"], 6)],
            ["Delta <= 0", f"{corr['Delta']['failure_count_Delta_le_zero']} / 200"],
        ], [7.0 * cm, 9.7 * cm]),
        figure("correlon_zero_delta_distribution.png", "図1. 確認的seedごとの保存-破壊分離。Correlon v1 は全seedでゼロ以下。", styles),
        PageBreak(),
        paragraph("3. 保存・破壊変換", styles, "heading"),
        paragraph(
            "Correlon v1 の平均Pは低く、最悪破壊条件での残存Dは1.0となった。すなわち、"
            "保存すべきものを安定に保持せず、破壊すべき条件で消えない。",
            styles,
        ),
        figure("correlon_zero_preservation_by_transform.png", "図2. P1-P7における方向性retention。0-to-0規約は図中に明記。", styles),
        figure("correlon_zero_destroy_residual_by_transform.png", "図3. D1-D8後の残存retention。高値は破壊後にも指標が残ることを表す。", styles),
        PageBreak(),
        paragraph("4. 表現変換の形式判定に関する留保", styles, "heading"),
        paragraph(
            "正式ラベルは事前登録の優先順位により FALSIFIED_REPRESENTATION_DEPENDENCE となった。"
            "ただしP1 node permutationとP4 orthogonal basis rotationの絶対スコア差の最大値は9.60e-15であり、"
            "1e-12の数値等価許容内である。形式的失敗率0.436667は、600 target-world中262件のbase scoreがゼロで、"
            "凍結式が unchanged 0-to-0 をretention 0と定義したことだけから生じた。",
            styles,
        ),
        table([
            ["観測", "値"],
            ["P1/P4最大絶対スコア差", "9.600e-15 (< 1e-12)"],
            ["ゼロ床target", f"{representation['target_base_zero_count']} / {representation['unique_target_world_count']}"],
            ["実証上の解釈", "観測された基底依存ではなく、target-score floor / sensitivity failure"],
        ], [6.1 * cm, 10.6 * cm]),
        paragraph(
            "この留保はv1を救済しない。以下の共通原因、matched null、ベースライン、敵対探索の失敗は、"
            "この形式ラベルとは独立にv1の必要条件を満たさないことを示す。",
            styles,
        ),
        figure("correlon_zero_preserve_destroy_plane.png", "図4. target世界とnull class centroidのPreserve-Destroy平面。", styles),
        PageBreak(),
        paragraph("5. 共通原因・matched null・ベースラインによる反証", styles, "heading"),
        table([
            ["反証", "観測値", "意味"],
            ["common_driver FPR", f"{common['crossing_count']}/{common['n']} = {fmt(common['false_positive_rate'], 3)}", "direct X-Y edgeなしで全件陽性"],
            ["matched_low_rank FPR", f"{low_rank['crossing_count']}/{low_rank['n']} = {fmt(low_rank['false_positive_rate'], 3)}", "低ランク共通因子だけで全件陽性"],
            ["最強baseline", baseline["strongest_baseline_by_mean_delta"], "mean Delta=" + fmt(baseline["strongest_baseline_mean_delta"], 6)],
            ["paired baseline advantage", fmt(baseline["paired_advantage"]["mean"], 6), "Correlonはper-seed最強baselineを大幅に下回る"],
        ], [4.3 * cm, 5.4 * cm, 7.0 * cm]),
        figure("correlon_zero_false_positive_rate.png", "図5. null class別false-positive rate。許容上限は0.05。", styles),
        figure("correlon_zero_vs_strongest_baseline.png", "図6. per-seedのCorrelon分離と最強baseline分離の比較。", styles),
        PageBreak(),
        paragraph("6. Anti-Correlon敵対探索", styles, "heading"),
        paragraph(
            "探索はlatent common factors、独立残差、drift、delayのみを使い、direct X-Y edgeを導入しない制約下で"
            "凍結スコアを最大化した。300 trial中82件が陽性閾値を越え、最大null scoreは0.743083だった。",
            styles,
        ),
        table([
            ["項目", "値"],
            ["凍結陽性閾値", fmt(audit["threshold"], 8)],
            ["最大adversarial null score", fmt(audit["maximum_score"], 6)],
            ["閾値超過", f"{audit['crossing_count']} / {audit['trials']} = {fmt(audit['crossing_rate'], 3)}"],
            ["best trial", str(audit["best_trial"])],
            ["best parameters", str(audit["best_parameters"])],
        ], [5.2 * cm, 11.5 * cm]),
        figure("correlon_zero_adversarial_trajectory.png", "図7. 敵対random searchのrunning maximum。閾値を初期に大幅超過。", styles),
        figure("correlon_zero_best_adversarial_example.png", "図8. 最良の許容敵対nullの観測主成分とlag cross-correlation。", styles),
        PageBreak(),
        paragraph("7. 最終決定と次実験", styles, "heading"),
        paragraph("最終決定: <b>FALSIFIED_REPRESENTATION_DEPENDENCE</b> (事前登録の形式的優先順位)。", styles),
        paragraph(
            "実証上、CorrelonZero_v1は target-score floor、common-driver false positive、matched-low-rank false positive、"
            "PCA rank-1 explained varianceへの劣位、反復可能なadversarial false positive により反証された。"
            "この結果はCorrelonの存在論を否定も肯定もしない。凍結した操作的定義v1が、テストした合成領域で"
            "transformation-selective relational invariantとして機能しなかったことだけを示す。",
            styles,
        ),
        paragraph("次実験", styles, "heading"),
        paragraph(
            "Correlon Zero v2を新規に事前登録する場合は、(1) untransformed target scoreが床にある候補を失格とする"
            "target-sensitivity gate、(2)不変性での明示的zero-to-zero equivalence ruleを先に凍結する。"
            "その後、fresh seedと未使用のgenerator familyを使い、今回のcommon-driver、matched-low-rank、"
            "adversarial parameter regionをholdoutとして最後に評価する。v1のseedでv2を調整してはならない。",
            styles,
        ),
        Spacer(1, 0.6 * cm),
        paragraph(
            "付記: 詳細な生CSV、JSON summary、図、Cemetery metadataは同一commitのresults/、figures/、cemetery/に保存される。"
            "このPDFはそれらの確定済み成果物を要約した提出用静的文書である。",
            styles,
        ),
    ]
    document = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4, leftMargin=1.7 * cm, rightMargin=1.7 * cm,
        topMargin=1.5 * cm, bottomMargin=1.8 * cm, title="Correlon Zero Final Report",
        author="osskosc-lab",
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


if __name__ == "__main__":
    build()
