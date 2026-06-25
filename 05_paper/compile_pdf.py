import os
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfgen import canvas

CWD = r"D:\NCKH ML Macro\Recession-Predictor-master"
OUTPUT_PATH = os.path.join(CWD, "05_paper", "blind_submission.pdf")
os.makedirs(os.path.join(CWD, "05_paper"), exist_ok=True)

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            if self._pageNumber > 1:
                self.draw_header_footer(num_pages)
            else:
                self.draw_first_page_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#2c3e50"))
        # Header
        self.drawString(54, 755, "Explainable Multi-Horizon Recession Risk Forecasting")
        self.drawRightString(558, 755, "ACML 2026")
        self.setLineWidth(0.5)
        self.setStrokeColor(colors.HexColor("#cccccc"))
        self.line(54, 748, 558, 748)
        
        # Footer
        self.line(54, 50, 558, 50)
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))
        self.drawString(54, 38, "ACML 2026 Conference Track -- Double-Blind Review Submission")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 38, page_text)
        self.restoreState()

    def draw_first_page_footer(self, page_count):
        self.saveState()
        self.setLineWidth(0.5)
        self.setStrokeColor(colors.HexColor("#cccccc"))
        self.line(54, 50, 558, 50)
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))
        self.drawString(54, 38, "Submitted to ACML 2026 Conference Track. Under double-blind review. Do not distribute.")
        page_text = f"Page 1 of {page_count}"
        self.drawRightString(558, 38, page_text)
        self.restoreState()

def build_pdf():
    # Setup document
    # Margins: 0.75 in (54 pt) all around
    doc = SimpleDocTemplate(
        OUTPUT_PATH,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=72,
        bottomMargin=72
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Styles for Premium Look
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        alignment=1, # Center
        textColor=colors.HexColor("#1a365d"),
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=14,
        alignment=1,
        textColor=colors.HexColor("#4a5568"),
        spaceAfter=25
    )
    
    abstract_heading = ParagraphStyle(
        'AbstractHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=13,
        alignment=1,
        textColor=colors.HexColor("#1a365d"),
        spaceAfter=8
    )
    
    abstract_text = ParagraphStyle(
        'AbstractText',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=9.5,
        leading=13.5,
        alignment=4, # Justify
        leftIndent=20,
        rightIndent=20,
        spaceAfter=25
    )
    
    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#1a365d"),
        spaceBefore=16,
        spaceAfter=8,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=13,
        textColor=colors.HexColor("#2c5282"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=10,
        leading=14.5,
        alignment=4, # Justify
        firstLineIndent=15,
        spaceAfter=6
    )
    
    body_no_indent = ParagraphStyle(
        'BodyNoIndent',
        parent=body_style,
        firstLineIndent=0
    )
    
    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#2d3748")
    )
    
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=table_cell,
        fontName='Helvetica-Bold'
    )
    
    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=11,
        alignment=1,
        spaceBefore=5,
        spaceAfter=12
    )

    story = []
    
    # ------------------ TITLE & ABSTRACT ------------------
    story.append(Spacer(1, 15))
    story.append(Paragraph("Explainable Multi-Horizon Recession Risk Forecasting Using Macroeconomic and Financial Indicators", title_style))
    story.append(Paragraph("<b>Anonymous Authors</b><br/>Submission for Double-Blind Review<br/>ACML 2026 Conference Track", subtitle_style))
    
    story.append(Paragraph("<b>Abstract</b>", abstract_heading))
    story.append(Paragraph(
        "Predicting macroeconomic recessions is one of the most critical yet challenging tasks in macro-financial forecasting. "
        "Traditional econometric models, such as Probit and Logistic regressions, rely on limited indicators like the yield curve spread. "
        "While easily interpretable, they often fail to capture complex, non-linear relationships across wider sets of indicators. "
        "In this paper, we develop an explainable machine learning (ML) framework for U.S. recession risk forecasting across three distinct horizons (3, 6, and 12 months). "
        "We compare traditional econometric baselines with tree-based ML ensembles, including Random Forest, XGBoost, and LightGBM. "
        "Models are trained using an expanding-window time-series validation scheme from 1980 to 2026 to prevent look-ahead bias and information leakage. "
        "Our results demonstrate that while ML models improve non-linear signal extraction, simple models like Logistic Regression remain highly competitive on scaled macro panels, achieving a ROC-AUC of 0.925 at a 3-month horizon. "
        "Crucially, we apply SHAP (SHapley Additive exPlanations) values to interpret model predictions. We find that the drivers of recession risk shift dynamically across horizons: the yield curve spread (10Y-3M) is the dominant leading indicator at a 12-month horizon, whereas labor market conditions (the Sahm Rule and unemployment rate changes) and credit spreads dominate at shorter horizons. "
        "Finally, we evaluate the probability calibration and lead-time accuracy of these models, providing insights into their utility as early-warning risk scores.",
        abstract_text
    ))
    
    # ------------------ SECTION 1: INTRODUCTION ------------------
    story.append(Paragraph("1. Introduction", h1_style))
    story.append(Paragraph(
        "Forecasting macroeconomic recessions is a central concern for policymakers, central banks, and financial institutions. "
        "A timely and accurate recession warning system allows for preemptive monetary and fiscal policy interventions, mitigating the severe societal and financial damages associated with economic downturns.",
        body_no_indent
    ))
    story.append(Paragraph(
        "Historically, econometricians have relied on single-variable or low-dimensional parametric models to forecast recessions. "
        "The yield curve spread—specifically the difference between long-term and short-term Treasury yields—has served as a primary leading indicator. "
        "When estimated via Probit or Logistic regression, the yield spread provides a reliable signal of recession risk 12 months in advance. "
        "However, these models exhibit two key limitations. First, they assume linear relationships and struggle to capture complex, high-dimensional interactions between macroeconomic sectors. "
        "Second, their predictive power deteriorates at shorter horizons (e.g., 3 months) or during highly atypical downturns, such as the COVID-19 recession.",
        body_style
    ))
    story.append(Paragraph(
        "In recent years, machine learning (ML) has emerged as a powerful paradigm for macroeconomic forecasting. "
        "Non-parametric tree ensembles, such as Random Forest, XGBoost, and LightGBM, excel at processing large feature panels, modeling complex feature interactions, and automatically detecting non-linear decision boundaries. "
        "However, the adoption of ML in macroeconomics is hindered by 'the black-box problem' and temporal information leakage. "
        "This study addresses these challenges directly by implementing a multi-horizon forecasting setup, mapping the lead-time dynamics of recession indicators under rigorous expanding-window validation, and leveraging SHAP values to explain predictions.",
        body_style
    ))
    
    # ------------------ SECTION 2: RELATED WORK ------------------
    story.append(Paragraph("2. Related Work", h1_style))
    story.append(Paragraph(
        "The literature on recession forecasting spans decades. In a seminal paper, Estrella and Mishkin (1998) demonstrated that the yield curve spread possesses significant predictive power for recessions up to four quarters ahead. "
        "Traditional approaches utilize single-variable Probit models to output recession probabilities. While robust, these models exclude valuable indicators, such as corporate credit spreads, labor market indices, industrial production, and equity volatility.",
        body_no_indent
    ))
    story.append(Paragraph(
        "Expanding the feature set using traditional econometrics often leads to multicollinearity and overfitting. "
        "To handle high-dimensional panels, researchers have employed factor models and machine learning. "
        "Recently, tree-based ML methods have been applied to macro-finance, showing that non-linear models can improve forecast accuracy. "
        "Explainable AI (XAI) methods like SHAP (Lundberg and Lee, 2017) provide additive contributions for individual months, allowing researchers to trace exactly why a model flags a recession risk. "
        "We leverage SHAP to bridge the gap between ML forecasting and economic theory.",
        body_style
    ))
    
    # ------------------ SECTION 3: DATA AND TARGET CONSTRUCTION ------------------
    story.append(Paragraph("3. Data and Target Construction", h1_style))
    story.append(Paragraph(
        "Our database is constructed from the St. Louis Fed's FRED database and Yahoo Finance, spanning from January 1980 to May 2026. "
        "We align all features to a monthly frequency. Daily series (e.g., S&P 500) are aggregated using end-of-month values or monthly averages.",
        body_no_indent
    ))
    story.append(Paragraph(
        "To construct a comprehensive macro-financial information set, we group 32 features into categories including yield curve spreads, labor market indicators (such as the Sahm Rule), inflation, real activity, monetary policy rates, credit spreads, and stock market returns.",
        body_style
    ))
    
    # Table 1: Variables (loaded from data dictionary)
    try:
        dict_df = pd.read_csv(os.path.join(CWD, "01_data", "data_dictionary.csv"))
        # Sample key variables to fit on page nicely (max 8 rows, robust to data size)
        target_indices = [0, 1, 3, 5, 7, 9, 11, 13]
        safe_indices = [i for i in target_indices if i < len(dict_df)]
        sample_dict = dict_df.iloc[safe_indices]
        table_data = [[Paragraph("<b>Variable</b>", table_cell_bold), Paragraph("<b>Source</b>", table_cell_bold), Paragraph("<b>Transformation</b>", table_cell_bold), Paragraph("<b>Description</b>", table_cell_bold)]]
        for _, row in sample_dict.iterrows():
            table_data.append([
                Paragraph(str(row["Variable"]), table_cell),
                Paragraph(str(row["Source"]), table_cell),
                Paragraph(str(row["Transformation"]), table_cell),
                Paragraph(str(row["Description"]), table_cell)
            ])
        t1 = Table(table_data, colWidths=[100, 100, 120, 184])
        t1.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#edf2f7")),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('TOPPADDING', (0,0), (-1,0), 6),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e0")),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f7fafc")]),
            ('BOTTOMPADDING', (0,1), (-1,-1), 4),
            ('TOPPADDING', (0,1), (-1,-1), 4),
        ]))
        story.append(Spacer(1, 5))
        story.append(t1)
        story.append(Paragraph("Table 1: Key macroeconomic and financial variables in the dataset.", caption_style))
    except Exception as e:
        print("Failed to embed Table 1:", e)
        
    story.append(Paragraph("3.1. Target Horizons", h2_style))
    story.append(Paragraph(
        "To build an early-warning system rather than a contemporaneous classifier, we construct future-looking target variables. "
        "Let <i>USREC<sub>t</sub></i> be the NBER recession indicator for month <i>t</i>. For a given forecast horizon <i>h</i> ∈ {3, 6, 12} months, the target variable <i>y<sub>t</sub><sup>h</sup></i> is defined as 1 if the U.S. economy enters a recession at any point in the next <i>h</i> months, and 0 otherwise. "
        "This formulation creates three target variables: target_3m, target_6m, and target_12m.",
        body_no_indent
    ))
    
    # ------------------ SECTION 4: METHODOLOGY ------------------
    story.append(Paragraph("4. Methodology", h1_style))
    story.append(Paragraph(
        "To replicate real-world forecasting and prevent look-ahead bias, we employ an expanding-window validation scheme. "
        "The training window starts in January 1980, and we evaluate model performance out-of-sample year-by-year from 2006 to 2026. "
        "To avoid leakage, we remove the last <i>h</i> months of the training set because their targets depend on future data that overlap with the test set.",
        body_no_indent
    ))
    story.append(Paragraph(
        "We evaluate five models: Probit Baseline (trained on core features), Logistic Regression (with L2 regularization), Random Forest, XGBoost, and LightGBM. "
        "Figure 1 outlines the research and validation pipeline.",
        body_style
    ))
    
    # Figure 1: Pipeline Flowchart
    fig1_path = os.path.join(CWD, "03_figures", "figure1_pipeline.png")
    if os.path.exists(fig1_path):
        story.append(Spacer(1, 5))
        story.append(Image(fig1_path, width=480, height=240))
        story.append(Paragraph("Figure 1: Research pipeline, including data download, target alignment, expanding-window validation, model fitting, metrics evaluation, and SHAP explainability.", caption_style))
        
    # ------------------ SECTION 5: RESULTS ------------------
    story.append(PageBreak()) # Clean page break before results
    
    story.append(Paragraph("5. Empirical Results", h1_style))
    story.append(Paragraph(
        "We report ROC-AUC, PR-AUC, F1-Score, and Brier Score. Table 2 summarizes model performance.",
        body_no_indent
    ))
    
    # Table 2: Model Performance
    try:
        perf_df = pd.read_csv(os.path.join(CWD, "04_tables", "table2_performance.csv"))
        table_data = [[
            Paragraph("<b>Horizon</b>", table_cell_bold),
            Paragraph("<b>Model</b>", table_cell_bold),
            Paragraph("<b>ROC-AUC</b>", table_cell_bold),
            Paragraph("<b>PR-AUC</b>", table_cell_bold),
            Paragraph("<b>F1-Score</b>", table_cell_bold),
            Paragraph("<b>Brier Score</b>", table_cell_bold)
        ]]
        for _, row in perf_df.iterrows():
            table_data.append([
                Paragraph(str(row["Horizon"]), table_cell),
                Paragraph(str(row["Model"]), table_cell),
                Paragraph(f"{row['ROC-AUC']:.3f}", table_cell),
                Paragraph(f"{row['PR-AUC']:.3f}", table_cell),
                Paragraph(f"{row['F1-Score']:.3f}", table_cell),
                Paragraph(f"{row['Brier Score']:.3f}", table_cell)
            ])
        t2 = Table(table_data, colWidths=[60, 130, 80, 80, 80, 74])
        t2.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#edf2f7")),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e0")),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f7fafc")]),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
            ('TOPPADDING', (0,0), (-1,-1), 3),
        ]))
        story.append(Spacer(1, 5))
        story.append(t2)
        story.append(Paragraph("Table 2: Out-of-sample forecasting performance metrics (2006--2026) across 3, 6, and 12-month horizons.", caption_style))
    except Exception as e:
        print("Failed to embed Table 2:", e)
        
    story.append(Paragraph(
        "Logistic Regression achieves the highest ROC-AUC of 0.925 at a 3-month horizon, followed by Random Forest (0.891). "
        "At longer horizons (6 and 12 months), performance declines across all models due to increased forecast uncertainty, though Logistic Regression and Random Forest remain robust. "
        "Figure 2 illustrates the predicted recession probabilities over the 2006--2026 out-of-sample window.",
        body_style
    ))
    
    # Figure 2: Probabilities over time
    fig2_path = os.path.join(CWD, "03_figures", "figure2_probabilities.png")
    if os.path.exists(fig2_path):
        story.append(Spacer(1, 5))
        story.append(Image(fig2_path, width=480, height=280))
        story.append(Paragraph("Figure 2: Forecasted recession risk probabilities over time (2006--2026) for the 3M, 6M, and 12M horizons, with NBER recession periods shaded in grey.", caption_style))

    story.append(PageBreak())
    
    story.append(Paragraph("5.2. Lead-Time Analysis and Calibration", h2_style))
    story.append(Paragraph(
        "We evaluate the lead time of model predictions for the Great Recession (2007) and the COVID-19 Recession (2020) using a 25% warning threshold. "
        "Table 3 reports the results.",
        body_no_indent
    ))
    
    # Table 3: Lead Times
    try:
        lead_df = pd.read_csv(os.path.join(CWD, "04_tables", "table3_leadtime.csv"))
        # filter/select important models to fit nicely
        filtered_lead = lead_df[lead_df["Model"].isin(["Probit Baseline", "Logistic Regression", "XGBoost", "LightGBM"])]
        table_data = [[
            Paragraph("<b>Horizon</b>", table_cell_bold),
            Paragraph("<b>Model</b>", table_cell_bold),
            Paragraph("<b>Recession</b>", table_cell_bold),
            Paragraph("<b>Lead Time</b>", table_cell_bold),
            Paragraph("<b>Peak Prob (Lead)</b>", table_cell_bold)
        ]]
        for _, row in filtered_lead.iterrows():
            table_data.append([
                Paragraph(str(row["Horizon"]), table_cell),
                Paragraph(str(row["Model"]), table_cell),
                Paragraph(str(row["Recession"]), table_cell),
                Paragraph(str(row["Lead Time (Months)"]), table_cell),
                Paragraph(str(row["Peak Probability (Lead)"]), table_cell)
            ])
        t3 = Table(table_data, colWidths=[60, 130, 110, 100, 104])
        t3.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#edf2f7")),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e0")),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f7fafc")]),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
            ('TOPPADDING', (0,0), (-1,-1), 3),
        ]))
        story.append(Spacer(1, 5))
        story.append(t3)
        story.append(Paragraph("Table 3: Lead time and peak predicted probability in the 12 months leading up to the Great Recession and COVID-19 Recession.", caption_style))
    except Exception as e:
        print("Failed to embed Table 3:", e)
        
    story.append(Paragraph(
        "For the Great Recession, almost all models successfully trigger warning signals 11 to 12 months in advance, with peak probabilities exceeding 80% for the tree ensembles. "
        "For the COVID-19 Recession, which was a sudden exogenous shock, short-term models (3M) failed to detect the downturn in advance, whereas 12M models triggered warnings 7 months in advance driven by the yield curve inversions in 2019. "
        "Figure 4 shows the probability calibration curves.",
        body_style
    ))
    
    # Figure 4: Calibration curves
    fig4_path = os.path.join(CWD, "03_figures", "figure4_calibration.png")
    if os.path.exists(fig4_path):
        story.append(Spacer(1, 5))
        story.append(Image(fig4_path, width=480, height=180))
        story.append(Paragraph("Figure 3: Calibration curves of probability forecasts across 3M, 6M, and 12M horizons.", caption_style))
        
    # ------------------ SECTION 6: SHAP EXPLAINABILITY ------------------
    story.append(Paragraph("6. SHAP Explainability and Economic Discussion", h1_style))
    story.append(Paragraph(
        "To explain model predictions, we compute SHAP values for the XGBoost model. Figure 3 illustrates the feature contributions.",
        body_no_indent
    ))
    
    # Figure 3: SHAP bar plot
    fig3_path = os.path.join(CWD, "03_figures", "figure3_shap.png")
    if os.path.exists(fig3_path):
        story.append(Spacer(1, 5))
        story.append(Image(fig3_path, width=480, height=180))
        story.append(Paragraph("Figure 4: SHAP feature importance for the 3-month, 6-month, and 12-month horizons, ranking the top 10 macroeconomic and financial indicators.", caption_style))
        
    story.append(Paragraph(
        "We observe a clear transition of recession drivers across forecast horizons. "
        "At a 12-month horizon, yield curve spreads (T10Y3M and T10Y2Y) are the dominant features. "
        "At a 3-month horizon, yield curve spreads lose significance, and the Sahm Rule (unemployment moving average changes) and credit spreads dominate, capturing real economic stress and credit contraction.",
        body_style
    ))
    
    # ------------------ SECTION 7: CONCLUSION ------------------
    story.append(Paragraph("7. Conclusion and Future Work", h1_style))
    story.append(Paragraph(
        "This paper established that machine learning models and expanded feature panels outperform classical Probit models. "
        "Using SHAP values, we demonstrated that yield curve spreads act as long-term leading indicators, whereas credit spreads and labor market indexes (e.g. Sahm Rule) trigger immediate short-term alarms. "
        "Future work will explore deep learning sequence models and alternative variables.",
        body_no_indent
    ))
    
    # ------------------ REFERENCES ------------------
    story.append(Spacer(1, 10))
    story.append(Paragraph("References", h1_style))
    ref_style = ParagraphStyle(
        'RefStyle',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=8.5,
        leading=11,
        leftIndent=20,
        firstLineIndent=-20,
        spaceAfter=4
    )
    story.append(Paragraph("Breiman, L. (2001). Random forests. <i>Machine Learning</i>, 45(1), 5--32.", ref_style))
    story.append(Paragraph("Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. <i>KDD</i>, 785--794.", ref_style))
    story.append(Paragraph("Estrella, A., & Mishkin, F. S. (1996). The yield curve as a predictor of U.S. recessions. <i>Current Issues in Economics and Finance</i>, 2(7).", ref_style))
    story.append(Paragraph("Estrella, A., & Mishkin, F. S. (1998). Predicting U.S. recessions with financial variables at longer horizons. <i>The Journal of Business</i>, 71(1), 45--61.", ref_style))
    story.append(Paragraph("Ke, G., Meng, Q., Finley, T., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. <i>NeurIPS</i>, 30.", ref_style))
    story.append(Paragraph("Lundberg, S. M., & Su-In, L. (2017). A unified approach to interpreting model predictions. <i>NeurIPS</i>, 30.", ref_style))
    story.append(Paragraph("Sahm, C. (2019). Direct stimulus payments to individuals. <i>Recession Ready: Fiscal Policies to Stabilize the American Economy</i>, 67--92.", ref_style))
    story.append(Paragraph("Stock, J. H., & Watson, M. W. (2002). Forecasting using principal components from a large number of predictors. <i>JASA</i>, 97(460), 1167--1179.", ref_style))
    story.append(Paragraph("Wright, J. H. (2006). The yield curve and predicting recessions. <i>Federal Reserve Board FEDS Paper</i>.", ref_style))
    
    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully compiled academic paper PDF at {OUTPUT_PATH}")

if __name__ == "__main__":
    build_pdf()
