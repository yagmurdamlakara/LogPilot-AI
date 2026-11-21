"""Streamlit dashboard for HDFS block-level anomaly detection."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.inference.predictor import LogPilotPredictor
from src.llm.analyzer import HDFSLogAnalyzer
from src.llm.client import OllamaError

DEFAULT_ARTIFACTS = ROOT / "model_artifacts" / "seed_11"


@st.cache_resource(show_spinner="LogPilot modeli yükleniyor...")
def load_predictor(artifact_dir: str) -> LogPilotPredictor:
    return LogPilotPredictor(artifact_dir)


@st.cache_resource(show_spinner="Yerel LLM analizcisi hazırlanıyor...")
def load_analyzer() -> HDFSLogAnalyzer:
    return HDFSLogAnalyzer()


def decoded_lines(uploaded_file):
    """Yield UTF-8 log lines from Streamlit's uploaded binary file."""
    for raw_line in uploaded_file:
        yield raw_line.decode("utf-8", errors="replace")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top:2.1rem;padding-bottom:3rem;max-width:1440px;}
        .lp-eyebrow {color:#8B9BB4;font-size:.76rem;letter-spacing:.16em;
                     font-weight:700;text-transform:uppercase;margin-bottom:.25rem;}
        .lp-subtitle {color:#A7B0BF;font-size:1rem;margin-top:-.45rem;}
        div[data-testid="stMetric"] {
            background:rgba(128,140,160,.08);
            border:1px solid rgba(128,140,160,.18);
            padding:1rem 1.1rem;border-radius:12px;
        }
        div[data-testid="stMetricLabel"] p {font-size:.86rem;}
        div[data-testid="stMetricValue"] {font-weight:700;}
        div[data-testid="stFileUploader"] {
            border:1px dashed rgba(128,150,180,.45);
            border-radius:12px;padding:.35rem;
        }
        .lp-note {color:#98A2B3;font-size:.82rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(predictions: pd.DataFrame) -> None:
    total = len(predictions)
    anomalies = int(predictions["prediction"].sum()) if total else 0
    normal = total - anomalies
    anomaly_rate = (100.0 * anomalies / total) if total else 0.0
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Analiz edilen blok", f"{total:,}")
    col2.metric("Anomali", f"{anomalies:,}")
    col3.metric("Normal", f"{normal:,}")
    col4.metric("Anomali oranı", f"%{anomaly_rate:.2f}")


def render_score_chart(predictions: pd.DataFrame) -> None:
    if predictions.empty:
        st.info("Grafik oluşturmak için tahmin sonucu bulunmuyor.")
        return

    scores = pd.to_numeric(predictions["anomaly_score"], errors="coerce").dropna()
    unique_scores = scores.nunique()
    if unique_scores <= 12:
        # Preserve exact score values for small/discrete distributions. This avoids
        # misleading auto-expanded histogram edges when every score is identical.
        distribution = scores.round(4).value_counts().sort_index()
        chart_data = distribution.rename_axis("Anomali skoru").to_frame("Blok sayısı")
        chart_data.index = chart_data.index.map(lambda value: f"{value:.4f}")
    else:
        bins = pd.cut(scores, bins=10, include_lowest=True)
        distribution = bins.value_counts(sort=False)
        chart_data = distribution.rename_axis("Skor aralığı").to_frame("Blok sayısı")
        chart_data.index = chart_data.index.astype(str)

    st.bar_chart(chart_data, y="Blok sayısı", use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="LogPilot AI | HDFS Log Intelligence",
        page_icon="🔎",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()

    st.markdown('<div class="lp-eyebrow">HDFS INTELLIGENT LOG ANALYTICS</div>',
                unsafe_allow_html=True)
    st.title("🔎 LogPilot AI")
    st.markdown(
        '<div class="lp-subtitle">Blok seviyesinde log analizi · '
        'Drain3 event parsing · Logistic Regression</div>',
        unsafe_allow_html=True,
    )
    st.write("HDFS log dosyanızı yükleyin; blokları analiz edin, şüpheli kayıtları "
             "inceleyin ve kararın dayandığı ham log kanıtlarına ulaşın.")
    st.divider()

    with st.sidebar:
        st.header("⚙️ Analiz ayarları")
        artifact_dir = st.text_input("Model paketi dizini", value=str(DEFAULT_ARTIFACTS))
        st.caption("Aktif model: Seed 11 · Logistic Regression")
        st.caption("Karar eşiği model paketindeki doğrulanmış validation eşiğidir.")
        st.divider()
        st.subheader("İşleme hattı")
        st.markdown(
            "1. HDFS log ayrıştırma\n"
            "2. Dondurulmuş Drain3 şablon eşleştirme\n"
            "3. Block bazlı olay sayımı\n"
            "4. Logistic Regression tahmini"
        )
        st.divider()
        st.caption("Tahmin sırasında Drain3 yeni event template öğrenmez.")

    try:
        predictor = load_predictor(str(Path(artifact_dir).expanduser().resolve()))
    except Exception as exc:
        st.error(f"Model paketi yüklenemedi: {exc}")
        st.info("Gerekirse python -m scripts.export_inference_bundle komutuyla "
                "model_artifacts/seed_11 paketi oluşturulmalıdır.")
        st.stop()

    st.subheader("📂 Log dosyası")
    uploaded_file = st.file_uploader(
        "HDFS log dosyası seçin",
        type=["log", "txt"],
        help="UTF-8 metin dosyaları desteklenir. Büyük dosyaların işlenmesi zaman alabilir.",
    )
    if uploaded_file is None:
        st.info("Başlamak için bir .log veya .txt dosyası yükleyin.")
        return

    st.caption(f"Seçilen dosya: **{uploaded_file.name}** · "
               f"{uploaded_file.size / (1024 * 1024):.2f} MB")
    analyze_col, clear_col = st.columns([3, 1])
    with analyze_col:
        run_analysis = st.button("▶  Logları analiz et", type="primary",
                                 use_container_width=True)
    with clear_col:
        clear_analysis = st.button("Sonuçları temizle", use_container_width=True)

    if clear_analysis:
        for key in ("logpilot_predictions", "logpilot_stats", "logpilot_filename",
                    "logpilot_evidence"):
            st.session_state.pop(key, None)
        st.rerun()

    if run_analysis:
        try:
            uploaded_file.seek(0)
            with st.spinner("Log satırları ayrıştırılıyor ve bloklar puanlanıyor..."):
                predictions, stats = predictor.predict_lines(decoded_lines(uploaded_file))
            st.session_state["logpilot_predictions"] = predictions
            st.session_state["logpilot_stats"] = stats
            st.session_state["logpilot_filename"] = uploaded_file.name
            st.session_state.pop("logpilot_evidence", None)
        except Exception as exc:
            st.exception(exc)
            st.stop()

    predictions = st.session_state.get("logpilot_predictions")
    if predictions is None:
        st.info("Analiz başlatıldığında özet, skor dağılımı ve block inceleme alanları burada görünecek.")
        return

    analyzed_filename = st.session_state.get("logpilot_filename", "")
    if analyzed_filename != uploaded_file.name:
        st.warning("Yeni bir dosya seçtiniz. Güncel sonuçları görmek için analizi yeniden çalıştırın.")
        return

    stats = st.session_state["logpilot_stats"]
    st.divider()
    st.subheader("📊 Analiz özeti")
    render_metrics(predictions)
    st.caption(
        f"Dosya: {analyzed_filename}  ·  Okunan satır: {stats['lines_seen']:,}  ·  "
        f"Block ID içeren satır: {stats['lines_with_block_ids']:,}  ·  "
        f"Eşleşmeyen satır: {stats['unmatched_lines']:,}  ·  "
        f"Karar eşiği: {predictor.threshold:.6f}"
    )

    overview_tab, anomalies_tab, evidence_tab, explanation_tab = st.tabs(
        ["Genel görünüm", "Anomali inceleme", "Ham log kanıtı", "LLM açıklama"]
    )

    with overview_tab:
        chart_col, context_col = st.columns([1.35, 1])
        with chart_col:
            st.markdown("#### Anomali skor dağılımı")
            render_score_chart(predictions)
            if not predictions.empty and predictions["anomaly_score"].nunique() == 1:
                only_score = float(predictions["anomaly_score"].iloc[0])
                st.warning(
                    f"Bu dosyadaki tüm block'lar aynı anomali skorunu ({only_score:.4f}) "
                    "aldı. Bu veri seti için skorlar arasında ayrım yapılamıyor; "
                    "bu sonucu tek başına model performansı olarak yorumlamayın."
                )
            elif predictions["prediction"].nunique() == 1:
                st.warning(
                    "Bu dosyada model yalnızca tek bir sınıf tahmin etti. "
                    "Sonucu veri setinin özelliklerini dikkate alarak yorumlayın."
                )
        with context_col:
            st.markdown("#### Sınıf dağılımı")
            counts = predictions["prediction_label"].value_counts()
            class_data = pd.DataFrame({
                "Blok sayısı": [int(counts.get("Normal", 0)),
                                int(counts.get("Anomaly", 0))]
            }, index=["Normal", "Anomali"])
            st.bar_chart(class_data, y="Blok sayısı", use_container_width=True)
            st.markdown("#### Model bilgisi")
            st.write("**Model:** Logistic Regression")
            st.write("**Seed:** 11")
            st.write(f"**Karar eşiği:** {predictor.threshold:.6f}")
            st.write("**Parser:** Train-only, frozen Drain3")
        st.info("Anomali etiketi modelin sınıflandırma kararıdır; tek başına kök nedeni kanıtlamaz.")

    anomalies = predictions.loc[predictions["prediction"] == 1].copy()
    with anomalies_tab:
        st.markdown("#### Tespit edilen anomaliler")
        if anomalies.empty:
            st.success("Bu dosyada seçilen karar eşiğini aşan blok bulunamadı.")
        else:
            filter_col, score_col = st.columns([1.5, 1])
            with filter_col:
                search_text = st.text_input("Block ID ara", placeholder="blk_...")
            with score_col:
                min_score = st.slider("Minimum anomali skoru", 0.0, 1.0, 0.0, 0.01)
            filtered = anomalies[anomalies["anomaly_score"] >= min_score].copy()
            if search_text.strip():
                filtered = filtered[
                    filtered["block_id"].astype(str).str.contains(
                        search_text.strip(), case=False, regex=False
                    )
                ]
            st.caption(f"{len(filtered):,} / {len(anomalies):,} anomali gösteriliyor.")
            st.dataframe(
                filtered,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "block_id": st.column_config.TextColumn("Block ID"),
                    "anomaly_score": st.column_config.ProgressColumn(
                        "Anomali skoru", min_value=0.0, max_value=1.0, format="%.4f"
                    ),
                    "prediction": None,
                    "prediction_label": st.column_config.TextColumn("Sınıf"),
                },
            )
            csv = anomalies.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇ Anomali listesini CSV indir", data=csv,
                file_name="logpilot_anomalies.csv", mime="text/csv",
            )
            all_csv = predictions.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇ Tüm block sonuçlarını CSV indir", data=all_csv,
                file_name="logpilot_all_predictions.csv", mime="text/csv",
            )

    with evidence_tab:
        st.markdown("#### Block bazında ham log inceleme")
        if anomalies.empty:
            st.info("İncelenecek anomali bulunmuyor.")
        else:
            evidence_options = anomalies["block_id"].astype(str).tolist()
            selected_block = st.selectbox(
                "Log kanıtını görüntülemek için block seçin",
                options=evidence_options,
                key="logpilot_selected_block",
            )
            if st.button("Seçili block'un log kanıtını getir", type="primary"):
                temporary_path = None
                try:
                    uploaded_file.seek(0)
                    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as temp:
                        temp.write(uploaded_file.getbuffer())
                        temporary_path = Path(temp.name)
                    evidence = predictor.extract_block_evidence(
                        temporary_path, {selected_block}, max_lines_per_block=100
                    )
                    st.session_state["logpilot_evidence"] = {
                        "block_id": selected_block,
                        "lines": evidence.get(selected_block, []),
                    }
                except Exception as exc:
                    st.error(f"Ham log kanıtı çıkarılamadı: {exc}")
                finally:
                    if temporary_path is not None:
                        temporary_path.unlink(missing_ok=True)

            evidence_state = st.session_state.get("logpilot_evidence")
            if evidence_state and evidence_state.get("block_id") == selected_block:
                st.markdown(f"**{selected_block} · Ham log kayıtları**")
                if evidence_state["lines"]:
                    st.caption(f"{len(evidence_state['lines'])} satır gösteriliyor (blok başına en fazla 100).")
                    st.code("\n".join(evidence_state["lines"]), language="text")
                else:
                    st.warning("Seçilen block için ham log satırı bulunamadı.")
            else:
                st.caption("Kanıtı görüntülemek için bir block seçip butona basın.")

    with explanation_tab:
        st.markdown("#### Seçilen anomali için LLM destekli açıklama")
        st.caption(
            "LLM yalnızca seçilen anormal bloğun ham log kayıtlarını inceler. "
            "Sınıflandırıcının tahminini veya anomali skorunu değiştirmez."
        )
        if anomalies.empty:
            st.info("Açıklama üretmek için önce en az bir anormal blok tespit edilmelidir.")
        else:
            explanation_block = st.selectbox(
                "Açıklanacak anormal blok",
                options=anomalies["block_id"].astype(str).tolist(),
                key="logpilot_explanation_block",
            )
            selected_prediction = anomalies.loc[
                anomalies["block_id"].astype(str) == explanation_block
            ].iloc[0]
            info_col, score_col = st.columns(2)
            info_col.metric("Sınıflandırıcı kararı", "Anomaly")
            score_col.metric("Anomali skoru", f"{float(selected_prediction['anomaly_score']):.4f}")
            if st.button("Seçili blok için açıklama üret", type="primary", key="logpilot_explain_button"):
                temporary_path = None
                try:
                    uploaded_file.seek(0)
                    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as temp:
                        temp.write(uploaded_file.getbuffer())
                        temporary_path = Path(temp.name)
                    with st.spinner("Seçilen bloğun kanıtları çıkarılıyor ve yerel LLM açıklama üretiyor..."):
                        evidence = predictor.extract_block_evidence(
                            temporary_path, {explanation_block}, max_lines_per_block=100
                        )
                        block_lines = evidence.get(explanation_block, [])
                        if not block_lines:
                            st.warning("Seçilen blok için ham log kanıtı bulunamadı; LLM çağrısı yapılmadı.")
                            st.session_state.pop("logpilot_explanation", None)
                        else:
                            analyzer = load_analyzer()
                            explanation = analyzer.explain_block(
                                block_id=explanation_block,
                                prediction=int(selected_prediction["prediction"]),
                                anomaly_score=float(selected_prediction["anomaly_score"]),
                                log_lines=block_lines,
                            )
                            st.session_state["logpilot_explanation"] = {
                                "filename": uploaded_file.name,
                                "block_id": explanation_block,
                                "result": explanation,
                            }
                except OllamaError as exc:
                    st.error(f"Yerel LLM bağlantı/yanıt hatası: {exc}")
                except Exception as exc:
                    st.error(f"LLM açıklaması oluşturulamadı: {exc}")
                finally:
                    if temporary_path is not None:
                        temporary_path.unlink(missing_ok=True)

            explanation_state = st.session_state.get("logpilot_explanation")
            if (
                explanation_state
                and explanation_state.get("filename") == uploaded_file.name
                and explanation_state.get("block_id") == explanation_block
            ):
                result = explanation_state["result"]
                st.divider()
                st.markdown("##### Açıklama özeti")
                st.write(result["summary"])
                st.markdown("##### Olası nedenler")
                for item in result["possible_causes"]:
                    st.markdown(f"- {item}")
                st.markdown("##### Log kanıtları")
                if result["evidence"]:
                    for line in result["evidence"]:
                        st.code(line, language="text")
                else:
                    st.info("LLM belirli bir kanıt satırı seçmedi.")
                st.markdown("##### Önerilen inceleme adımları")
                for item in result["recommendations"]:
                    st.markdown(f"- {item}")
                if result["limitations"]:
                    st.markdown("##### Sınırlılıklar")
                    for item in result["limitations"]:
                        st.markdown(f"- {item}")
                confidence_labels = {"düşük": "Düşük", "orta": "Orta", "yüksek": "Yüksek"}
                st.metric("Kanıta göre sınırlandırılmış güven seviyesi", confidence_labels[result["confidence"]])
                if result["model_confidence"] != result["confidence"]:
                    st.caption(
                        f"Modelin ham güven beyanı: {confidence_labels[result['model_confidence']]}. "
                        "Gösterilen seviye, seçilen kanıt sayısına göre daha ihtiyatlı hale getirildi."
                    )
                st.warning(
                    "Bu açıklama olası nedenleri ve inceleme adımlarını sunar; "
                    "HDFS normal/anormal etiketi gerçek kök nedeni kanıtlamaz."
                )

    st.divider()
    st.markdown(
        '<div class="lp-note">LogPilot AI · HDFS block-level anomaly detection · '
        'Model tahmini ve log kanıtı ayrı değerlendirilmelidir.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
