"""Streamlit interface for HDFS block-level anomaly detection."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Streamlit may put src/app, rather than the repository root, on sys.path.
# Add the repository root so imports such as src.inference resolve reliably.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.inference.predictor import LogPilotPredictor

DEFAULT_ARTIFACTS = ROOT / "model_artifacts" / "seed_11"


@st.cache_resource(show_spinner="LogPilot modeli yükleniyor...")
def load_predictor(artifact_dir: str) -> LogPilotPredictor:
    return LogPilotPredictor(artifact_dir)


def decoded_lines(uploaded_file):
    """Yield UTF-8 log lines from Streamlit's uploaded binary file."""
    for raw_line in uploaded_file:
        yield raw_line.decode("utf-8", errors="replace")


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


def main() -> None:
    st.set_page_config(
        page_title="LogPilot AI | HDFS Log Analizi",
        page_icon="🔎",
        layout="wide",
    )
    st.title("🔎 LogPilot AI")
    st.caption("HDFS blok seviyesinde log anomalisi tespiti")
    st.write(
        "Bir HDFS log dosyası yükleyin. LogPilot, Drain3 ile olay şablonlarını "
        "eşleştirir ve her blok için Logistic Regression anomali skoru üretir."
    )

    with st.sidebar:
        st.header("Model bilgisi")
        artifact_dir = st.text_input(
            "Model paketi dizini", value=str(DEFAULT_ARTIFACTS)
        )
        st.caption("Varsayılan model: Seed 11 Logistic Regression")
        st.divider()
        st.caption(
            "Model, yalnızca eğitim bloklarıyla oluşturulmuş sabit Drain3 "
            "şablonlarını kullanır. Tahmin sırasında yeni şablon öğrenmez."
        )

    try:
        predictor = load_predictor(str(Path(artifact_dir).expanduser().resolve()))
    except Exception as exc:
        st.error(f"Model paketi yüklenemedi: {exc}")
        st.info(
            "Önce `python -m scripts.export_inference_bundle` komutuyla "
            "model_artifacts/seed_11 paketini oluşturun."
        )
        st.stop()

    uploaded_file = st.file_uploader(
        "HDFS log dosyası seçin",
        type=["log", "txt"],
        help="Dosya UTF-8 olarak okunur; çözümlenemeyen karakterler değiştirilir.",
    )
    if uploaded_file is None:
        st.info("Analize başlamak için bir HDFS log dosyası yükleyin.")
        return

    st.caption(
        f"Dosya: {uploaded_file.name} · "
        f"{uploaded_file.size / (1024 * 1024):.2f} MB"
    )
    if st.button("Logları analiz et", type="primary", use_container_width=True):
        try:
            uploaded_file.seek(0)
            with st.spinner("Log satırları işleniyor ve bloklar puanlanıyor..."):
                predictions, stats = predictor.predict_lines(
                    decoded_lines(uploaded_file)
                )
            st.session_state["logpilot_predictions"] = predictions
            st.session_state["logpilot_stats"] = stats
            st.session_state["logpilot_filename"] = uploaded_file.name
            st.session_state.pop("logpilot_evidence", None)
        except Exception as exc:
            st.exception(exc)
            st.stop()

    predictions = st.session_state.get("logpilot_predictions")
    if predictions is None:
        return

    stats = st.session_state["logpilot_stats"]
    st.subheader("Analiz özeti")
    render_metrics(predictions)
    st.caption(
        f"Okunan satır: {stats['lines_seen']:,} · "
        f"Block ID içeren satır: {stats['lines_with_block_ids']:,} · "
        f"Eşleşmeyen satır: {stats['unmatched_lines']:,} · "
        f"Karar eşiği: {predictor.threshold:.6f}"
    )

    anomalies = predictions.loc[predictions["prediction"] == 1].copy()
    st.subheader("Tespit edilen anomaliler")
    if anomalies.empty:
        st.success("Bu dosyada seçilen karar eşiğini aşan blok bulunamadı.")
    else:
        st.dataframe(
            anomalies,
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
            "Anomali listesini CSV indir",
            data=csv,
            file_name="logpilot_anomalies.csv",
            mime="text/csv",
        )

        selected_block = st.selectbox(
            "Log kanıtını görüntülemek için blok seçin",
            options=anomalies["block_id"].astype(str).tolist(),
        )
        if st.button("Seçili bloğun log kanıtını göster"):
            uploaded_file.seek(0)
            with tempfile.NamedTemporaryFile(
                suffix=".log", delete=True
            ) as temporary_log:
                temporary_log.write(uploaded_file.getbuffer())
                temporary_log.flush()
                evidence = predictor.extract_block_evidence(
                    temporary_log.name, {selected_block}, max_lines_per_block=100
                )
            st.session_state["logpilot_evidence"] = {
                "block_id": selected_block,
                "lines": evidence.get(selected_block, []),
            }

        evidence = st.session_state.get("logpilot_evidence")
        if evidence:
            st.markdown(f"**{evidence['block_id']} için ham log kanıtı**")
            if evidence["lines"]:
                st.code("\n".join(evidence["lines"]), language="text")
            else:
                st.warning("Seçilen blok için ham log satırı bulunamadı.")

    st.caption(
        "Not: Anomali tespiti bir sınıflandırma sonucudur; tek başına kök nedeni "
        "kanıtlamaz. Açıklamalar, ilerleyen aşamada seçilen bloğun log kanıtına "
        "dayalı olarak üretilecektir."
    )


if __name__ == "__main__":
    main()
