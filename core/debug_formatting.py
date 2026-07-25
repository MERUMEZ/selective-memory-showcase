"""
================================================================================
 DEBUG_FORMATTING.PY — Общий форматтер системного дебаг-блока "мозга"
================================================================================
Вынесено из main.py, чтобы ОДНА и та же логика форматирования дебаг-блока
использовалась и консольным CLI (main.py), и Telegram-ботом (core/brain_session.py)
без дублирования кода. Здесь только ФОРМАТИРОВАНИЕ (возврат строки) — печать
в консоль/отправку в Telegram делает вызывающий код.
================================================================================
"""

from typing import Optional


def format_debug_block(
    brain_time: float,
    session_elapsed: float,
    emotion_score: float,
    perplexity: float,
    total_density: float,
    confidence: float,
    stress_state,
    spike_triggered: bool,
    memory_written: bool,
    response_source: str,
    decayed_nodes: int,
    total_nodes: int,
    top_nodes,
    stm_status: str,
    consolidation_event: Optional[str] = None,
    prompt_context: Optional[str] = None,
    reward_trace: Optional[str] = None,
    reward_eval: Optional[str] = None,
    activation_traces: Optional[list] = None,
    mood_state=None,
) -> str:
    """
    Формирует единый системный дебаг-блок как строку (многострочную).
    Сигнатура идентична бывшему main.py:print_debug_block — только
    вместо print() построчно, здесь всё собирается в список строк и
    склеивается в конце через "\n".
    """
    lines = []
    lines.append("┌──────────────────── [BRAIN DEBUG] ────────────────────")
    lines.append(f"│ Brain time (t)          : {brain_time:.1f}s (session +{session_elapsed:.1f}s)")
    lines.append(f"│ Emotion score           : {emotion_score:.3f}")
    lines.append(f"│ Perplexity              : {perplexity:.3f}")
    lines.append(f"│ Total density (spike)   : {total_density:.3f}")
    lines.append(f"│ Cortex confidence       : {confidence:.3f}")
    if mood_state is not None:
        lines.append(
            f"│ Mood vector             : joy={mood_state.joy:.2f} "
            f"anxiety={mood_state.anxiety:.2f} curiosity={mood_state.curiosity:.2f} "
            f"affection={mood_state.affection:.2f} (dominant={mood_state.dominant_emotion()})"
        )
    lines.append(f"│ Response source         : {response_source}")
    lines.append(f"│ Current stress          : {stress_state.current_stress:.3f}")
    lines.append(f"│ Dynamic spike threshold : {stress_state.effective_plasticity_threshold:.3f}")
    lines.append(f"│ Stress overloaded?      : {stress_state.is_overloaded}")
    lines.append(f"│ Spike triggered?        : {spike_triggered}")
    lines.append(f"│ Written to DB?          : {memory_written}")
    lines.append(f"│ Decay applied to        : {decayed_nodes} / {total_nodes} nodes (this tick)")
    lines.append(f"│ Working memory (STM)    : {stm_status}")
    if consolidation_event:
        lines.append(f"│ Consolidation event     : {consolidation_event}")
    if reward_trace:
        lines.append(f"│ {reward_trace}")
    if reward_eval:
        lines.append(f"│ {reward_eval}")
    if activation_traces:
        lines.append("│ Active associations (Spreading Activation):")
        for trace in activation_traces:
            lines.append(
                f"│   [ASSOCIATION] Node {trace.source_id} -> Node {trace.target_id} "
                f"(edge_weight={trace.edge_weight:.2f}, activation_score={trace.activation_score:.3f})"
            )
    lines.append("│ Top memory nodes (by weight):")
    if top_nodes:
        for node in top_nodes:
            preview = node.context.strip().replace("\n", " ")[:35]
            lines.append(f"│   id={node.id:<4} weight={node.weight:.3f}  {preview!r}")
    else:
        lines.append("│   (память пуста)")

    prompt_context_display = (
        prompt_context.replace("\n", " | ") if prompt_context else "None"
    )
    lines.append(f"│ Prompt context          : {prompt_context_display}")
    lines.append("└─────────────────────────────────────────────────────────")

    return "\n".join(lines)