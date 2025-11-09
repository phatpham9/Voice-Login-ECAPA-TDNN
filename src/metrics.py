"""
Performance Metrics Dashboard for Voice Authentication System

This module provides tools to calculate and visualize key performance metrics:
- FAR (False Acceptance Rate): Rate of impostor acceptance
- FRR (False Rejection Rate): Rate of genuine user rejection
- EER (Equal Error Rate): Point where FAR = FRR
- ROC Curve (Receiver Operating Characteristic)
- DET Curve (Detection Error Tradeoff)
- Score distributions for genuine vs impostor attempts
"""

import numpy as np
from typing import List, Dict, Optional
import plotly.graph_objects as go
from src.database import get_auth_history, list_users
from scipy.interpolate import interp1d
from scipy.optimize import brentq


def calculate_far_frr(
    genuine_scores: List[float],
    impostor_scores: List[float],
    thresholds: Optional[List[float]] = None,
) -> Dict:
    """
    Calculate False Acceptance Rate (FAR) and False Rejection Rate (FRR)

    Args:
        genuine_scores: List of similarity scores for genuine user attempts
        impostor_scores: List of similarity scores for impostor attempts
        thresholds: List of threshold values to evaluate (default: linspace from 0 to 1)

    Returns:
        Dictionary containing:
        - thresholds: List of threshold values
        - far: False Acceptance Rate at each threshold
        - frr: False Rejection Rate at each threshold
        - eer: Equal Error Rate (where FAR = FRR)
        - eer_threshold: Threshold value at EER
    """
    if not genuine_scores or not impostor_scores:
        return {
            "thresholds": [],
            "far": [],
            "frr": [],
            "eer": 0.0,
            "eer_threshold": 0.0,
            "error": "Insufficient data for metrics calculation",
        }

    if thresholds is None:
        thresholds = np.linspace(0, 1, 1000)

    far_list = []
    frr_list = []

    genuine_scores_arr = np.array(genuine_scores)
    impostor_scores_arr = np.array(impostor_scores)

    for threshold in thresholds:
        # FAR: Percentage of impostors incorrectly accepted (score >= threshold)
        false_accepts = np.sum(impostor_scores_arr >= threshold)
        far = false_accepts / len(impostor_scores_arr)
        far_list.append(far)

        # FRR: Percentage of genuine users incorrectly rejected (score < threshold)
        false_rejects = np.sum(genuine_scores_arr < threshold)
        frr = false_rejects / len(genuine_scores_arr)
        frr_list.append(frr)

    far_list = np.array(far_list)
    frr_list = np.array(frr_list)

    # Calculate EER (Equal Error Rate) - where FAR = FRR
    try:
        # Find threshold where FAR = FRR using interpolation
        far_interp = interp1d(thresholds, far_list, kind="linear")
        frr_interp = interp1d(thresholds, frr_list, kind="linear")

        # Find where FAR(t) - FRR(t) = 0
        eer_threshold = brentq(
            lambda t: far_interp(t) - frr_interp(t), thresholds[0], thresholds[-1]
        )
        eer = float(far_interp(eer_threshold))
    except Exception as e:
        # Fallback: find closest point
        diff = np.abs(far_list - frr_list)
        eer_idx = np.argmin(diff)
        eer_threshold = thresholds[eer_idx]
        eer = (far_list[eer_idx] + frr_list[eer_idx]) / 2

    return {
        "thresholds": thresholds.tolist(),
        "far": far_list.tolist(),
        "frr": frr_list.tolist(),
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
    }


def generate_roc_curve(genuine_scores: List[float], impostor_scores: List[float]):
    """
    Generate ROC (Receiver Operating Characteristic) curve

    ROC plots True Positive Rate (TPR) vs False Positive Rate (FPR)
    - TPR = 1 - FRR (genuine acceptance rate)
    - FPR = FAR (impostor acceptance rate)

    Returns:
        Plotly figure object
    """
    if not genuine_scores or not impostor_scores:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for ROC curve",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    metrics = calculate_far_frr(genuine_scores, impostor_scores)

    far = np.array(metrics["far"])
    frr = np.array(metrics["frr"])
    tpr = 1 - frr  # True Positive Rate = 1 - False Rejection Rate
    fpr = far  # False Positive Rate = False Acceptance Rate

    # Calculate AUC (Area Under Curve) using trapezoidal rule
    auc = np.trapz(tpr, fpr)

    fig = go.Figure()

    # ROC curve
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC Curve (AUC={auc:.4f})",
            line=dict(color="blue", width=2),
        )
    )

    # Diagonal reference line (random classifier)
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Classifier",
            line=dict(color="gray", dash="dash", width=1),
        )
    )

    # Mark EER point
    eer_idx = np.argmin(np.abs(far - frr))
    fig.add_trace(
        go.Scatter(
            x=[fpr[eer_idx]],
            y=[tpr[eer_idx]],
            mode="markers",
            name=f"EER Point ({metrics['eer']:.3f})",
            marker=dict(color="red", size=10, symbol="diamond"),
        )
    )

    fig.update_layout(
        title="ROC Curve (Receiver Operating Characteristic)",
        xaxis_title="False Positive Rate (FAR)",
        yaxis_title="True Positive Rate (1 - FRR)",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        width=700,
        height=600,
        showlegend=True,
    )

    return fig


def generate_det_curve(genuine_scores: List[float], impostor_scores: List[float]):
    """
    Generate DET (Detection Error Tradeoff) curve

    DET plots False Rejection Rate (FRR) vs False Acceptance Rate (FAR)
    Uses normal deviate scale for better visualization of low error rates

    Returns:
        Plotly figure object
    """
    if not genuine_scores or not impostor_scores:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for DET curve",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    metrics = calculate_far_frr(genuine_scores, impostor_scores)

    far = np.array(metrics["far"])
    frr = np.array(metrics["frr"])

    # Clip to avoid log(0)
    far_clipped = np.clip(far, 1e-6, 1 - 1e-6)
    frr_clipped = np.clip(frr, 1e-6, 1 - 1e-6)

    fig = go.Figure()

    # DET curve
    fig.add_trace(
        go.Scatter(
            x=far * 100,  # Convert to percentage
            y=frr * 100,
            mode="lines",
            name="DET Curve",
            line=dict(color="green", width=2),
        )
    )

    # Diagonal reference line (FAR = FRR)
    max_val = max(np.max(far * 100), np.max(frr * 100))
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            name="FAR = FRR",
            line=dict(color="gray", dash="dash", width=1),
        )
    )

    # Mark EER point
    eer_idx = np.argmin(np.abs(far - frr))
    fig.add_trace(
        go.Scatter(
            x=[far[eer_idx] * 100],
            y=[frr[eer_idx] * 100],
            mode="markers",
            name=f"EER Point ({metrics['eer']*100:.2f}%)",
            marker=dict(color="red", size=10, symbol="diamond"),
        )
    )

    fig.update_layout(
        title="DET Curve (Detection Error Tradeoff)",
        xaxis_title="False Acceptance Rate (FAR) [%]",
        yaxis_title="False Rejection Rate (FRR) [%]",
        xaxis_type="log",
        yaxis_type="log",
        width=700,
        height=600,
        showlegend=True,
    )

    return fig


def generate_score_distribution(
    genuine_scores: List[float], impostor_scores: List[float], current_threshold: float
):
    """
    Generate histogram of score distributions for genuine vs impostor attempts

    Args:
        genuine_scores: List of similarity scores for genuine users
        impostor_scores: List of similarity scores for impostors
        current_threshold: Current threshold value to display

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    if genuine_scores:
        fig.add_trace(
            go.Histogram(
                x=genuine_scores,
                name="Genuine Users",
                opacity=0.7,
                marker_color="green",
                nbinsx=50,
            )
        )

    if impostor_scores:
        fig.add_trace(
            go.Histogram(
                x=impostor_scores,
                name="Impostors",
                opacity=0.7,
                marker_color="red",
                nbinsx=50,
            )
        )

    # Add threshold line
    fig.add_vline(
        x=current_threshold,
        line_dash="dash",
        line_color="blue",
        annotation_text=f"Threshold ({current_threshold:.2f})",
        annotation_position="top",
    )

    fig.update_layout(
        title="Score Distribution: Genuine vs Impostor Attempts",
        xaxis_title="Similarity Score",
        yaxis_title="Count",
        barmode="overlay",
        width=800,
        height=500,
        showlegend=True,
    )

    return fig


def generate_far_frr_curve(genuine_scores: List[float], impostor_scores: List[float]):
    """
    Generate FAR/FRR vs Threshold curve

    Shows how FAR and FRR change with different threshold values

    Returns:
        Plotly figure object
    """
    if not genuine_scores or not impostor_scores:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for FAR/FRR curve",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    metrics = calculate_far_frr(genuine_scores, impostor_scores)

    thresholds = np.array(metrics["thresholds"])
    far = np.array(metrics["far"])
    frr = np.array(metrics["frr"])

    fig = go.Figure()

    # FAR curve
    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=far * 100,
            mode="lines",
            name="FAR (False Acceptance)",
            line=dict(color="red", width=2),
        )
    )

    # FRR curve
    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=frr * 100,
            mode="lines",
            name="FRR (False Rejection)",
            line=dict(color="blue", width=2),
        )
    )

    # Mark EER point
    fig.add_trace(
        go.Scatter(
            x=[metrics["eer_threshold"]],
            y=[metrics["eer"] * 100],
            mode="markers",
            name=f"EER ({metrics['eer']*100:.2f}%)",
            marker=dict(color="green", size=10, symbol="diamond"),
        )
    )

    # Add vertical line at EER threshold
    fig.add_vline(
        x=metrics["eer_threshold"],
        line_dash="dash",
        line_color="green",
        annotation_text=f"EER Threshold ({metrics['eer_threshold']:.3f})",
        annotation_position="top",
    )

    fig.update_layout(
        title="FAR and FRR vs Threshold",
        xaxis_title="Threshold",
        yaxis_title="Error Rate [%]",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 100]),
        width=800,
        height=600,
        showlegend=True,
    )

    return fig


def collect_test_data() -> Dict:
    """
    Collect authentication data from the database and classify as genuine or impostor

    Returns:
        Dictionary containing:
        - genuine_scores: List of scores from genuine user attempts
        - impostor_scores: List of scores from impostor attempts
        - total_attempts: Total number of authentication attempts
    """
    auth_history = get_auth_history(limit=10000)  # Get all history

    if not auth_history:
        return {
            "genuine_scores": [],
            "impostor_scores": [],
            "total_attempts": 0,
            "genuine_attempts": 0,
            "impostor_attempts": 0,
        }

    genuine_scores = []
    impostor_scores = []

    for record in auth_history:
        username = record["username"]
        score = record["score"]
        success = record["success"]

        # Check if user exists in database
        user_exists = username in list_users()

        if user_exists and success:
            # Successful authentication - likely genuine
            genuine_scores.append(score)
        elif user_exists and not success:
            # Failed authentication of enrolled user
            # Could be genuine (low quality audio) or impostor
            # We'll classify based on score: high score = genuine with bad threshold
            if score > 0.65:  # Reasonably high score, probably genuine
                genuine_scores.append(score)
            else:
                impostor_scores.append(score)
        else:
            # User doesn't exist or failed - likely impostor
            impostor_scores.append(score)

    return {
        "genuine_scores": genuine_scores,
        "impostor_scores": impostor_scores,
        "total_attempts": len(auth_history),
        "genuine_attempts": len(genuine_scores),
        "impostor_attempts": len(impostor_scores),
    }


def generate_metrics_summary(
    genuine_scores: List[float], impostor_scores: List[float], current_threshold: float
) -> str:
    """
    Generate a text summary of performance metrics

    Args:
        genuine_scores: List of similarity scores for genuine users
        impostor_scores: List of similarity scores for impostors
        current_threshold: Current threshold value

    Returns:
        Formatted string with metrics summary
    """
    if not genuine_scores or not impostor_scores:
        return """⚠️ **Insufficient Data for Metrics**

To generate meaningful performance metrics, the system needs:
- Multiple authentication attempts from enrolled users (genuine attempts)
- Multiple authentication attempts from non-enrolled users or failed attempts (impostor attempts)

**Current Data:**
- Genuine attempts: {0}
- Impostor attempts: {1}

**Recommendations:**
1. Enroll multiple users with 2-3 voice samples each
2. Perform successful login attempts for each user
3. Perform intentional failed login attempts (wrong user/voice)
4. Come back to this dashboard after collecting more data""".format(
            len(genuine_scores), len(impostor_scores)
        )

    metrics = calculate_far_frr(genuine_scores, impostor_scores)

    # Calculate metrics at current threshold
    current_far = (
        np.sum(np.array(impostor_scores) >= current_threshold) / len(impostor_scores)
        if impostor_scores
        else 0.0
    )
    current_frr = (
        np.sum(np.array(genuine_scores) < current_threshold) / len(genuine_scores)
        if genuine_scores
        else 0.0
    )

    # Calculate statistics
    genuine_mean = np.mean(genuine_scores) if genuine_scores else 0
    genuine_std = np.std(genuine_scores) if genuine_scores else 0
    impostor_mean = np.mean(impostor_scores) if impostor_scores else 0
    impostor_std = np.std(impostor_scores) if impostor_scores else 0

    summary = f"""## 📊 Performance Metrics Summary

### Overall Statistics
- **Total Genuine Attempts:** {len(genuine_scores)}
- **Total Impostor Attempts:** {len(impostor_scores)}
- **Total Attempts:** {len(genuine_scores) + len(impostor_scores)}

### Score Distributions
- **Genuine Score:** {genuine_mean:.3f} ± {genuine_std:.3f}
- **Impostor Score:** {impostor_mean:.3f} ± {impostor_std:.3f}
- **Separation:** {genuine_mean - impostor_mean:.3f} (higher is better)

### Current Threshold Performance ({current_threshold:.2f})
- **FAR (False Acceptance Rate):** {current_far*100:.2f}% - {len([s for s in impostor_scores if s >= current_threshold])}/{len(impostor_scores)} impostors accepted
- **FRR (False Rejection Rate):** {current_frr*100:.2f}% - {len([s for s in genuine_scores if s < current_threshold])}/{len(genuine_scores)} genuine users rejected

### Optimal Performance (EER Point)
- **EER (Equal Error Rate):** {metrics['eer']*100:.2f}%
- **EER Threshold:** {metrics['eer_threshold']:.3f}
- **Interpretation:** At threshold {metrics['eer_threshold']:.3f}, both FAR and FRR are {metrics['eer']*100:.2f}%

### Recommendations
"""

    # Add recommendations based on metrics
    if metrics["eer"] < 0.05:
        summary += "✅ **Excellent performance!** EER < 5% indicates a highly accurate system.\n"
    elif metrics["eer"] < 0.10:
        summary += (
            "✅ **Good performance.** EER < 10% is acceptable for most applications.\n"
        )
    elif metrics["eer"] < 0.20:
        summary += "⚠️ **Moderate performance.** Consider collecting more/better quality samples.\n"
    else:
        summary += "❌ **Poor performance.** System needs improvement. Check audio quality and sample diversity.\n"

    if current_far > 0.10:
        summary += f"⚠️ **Security Warning:** Current FAR ({current_far*100:.1f}%) is high. Consider increasing threshold to {metrics['eer_threshold']:.2f}\n"

    if current_frr > 0.20:
        summary += f"⚠️ **Usability Warning:** Current FRR ({current_frr*100:.1f}%) is high. Users are frequently rejected. Consider lowering threshold to {metrics['eer_threshold']:.2f}\n"

    if abs(current_threshold - metrics["eer_threshold"]) > 0.10:
        summary += f"💡 **Optimization:** Your current threshold ({current_threshold:.2f}) differs significantly from optimal EER threshold ({metrics['eer_threshold']:.3f}). Consider adjusting.\n"

    return summary


def generate_confusion_matrix(
    genuine_scores: List[float], impostor_scores: List[float], threshold: float
):
    """
    Generate confusion matrix visualization

    Args:
        genuine_scores: List of similarity scores for genuine users
        impostor_scores: List of similarity scores for impostors
        threshold: Threshold value for classification

    Returns:
        Plotly figure object
    """
    if not genuine_scores or not impostor_scores:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for confusion matrix",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    # Calculate confusion matrix values
    genuine_arr = np.array(genuine_scores)
    impostor_arr = np.array(impostor_scores)

    true_positives = np.sum(genuine_arr >= threshold)  # Genuine correctly accepted
    false_negatives = np.sum(genuine_arr < threshold)  # Genuine incorrectly rejected
    false_positives = np.sum(impostor_arr >= threshold)  # Impostor incorrectly accepted
    true_negatives = np.sum(impostor_arr < threshold)  # Impostor correctly rejected

    # Create confusion matrix
    confusion_matrix = np.array(
        [[true_positives, false_negatives], [false_positives, true_negatives]]
    )

    labels = [
        [f"TP: {true_positives}", f"FN: {false_negatives}"],
        [f"FP: {false_positives}", f"TN: {true_negatives}"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=confusion_matrix,
            x=["Accepted", "Rejected"],
            y=["Genuine User", "Impostor"],
            text=labels,
            texttemplate="%{text}",
            textfont={"size": 16},
            colorscale="Blues",
            showscale=True,
        )
    )

    fig.update_layout(
        title=f"Confusion Matrix (Threshold: {threshold:.2f})",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        width=600,
        height=500,
    )

    return fig
