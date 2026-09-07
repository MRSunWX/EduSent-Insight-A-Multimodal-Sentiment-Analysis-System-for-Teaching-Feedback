from __future__ import annotations

import io
import math
import re
import time
import uuid
from dataclasses import dataclass

from PIL import Image, ImageStat

from .schemas import (
    AnalysisResponse,
    AspectResult,
    ConsistencyResult,
    EmotionScore,
    ImageEvidence,
    TextEvidence,
)


POSITIVE_TERMS = {
    "太棒了": 1.8,
    "太好了": 1.8,
    "真好": 1.4,
    "挺好": 1.2,
    "深入浅出": 1.8,
    "清楚": 1.4,
    "清晰": 1.4,
    "启发": 1.5,
    "有趣": 1.3,
    "喜欢": 1.5,
    "满意": 1.6,
    "很好": 1.7,
    "不错": 1.3,
    "生动": 1.4,
    "耐心": 1.2,
    "及时": 1.1,
    "实用": 1.3,
    "丰富": 1.1,
    "容易理解": 1.6,
    "收获很大": 1.8,
    "能帮助": 1.3,
    "巩固知识": 1.5,
    "有挑战性": 0.7,
}

NEGATIVE_TERMS = {
    "也就那样": 1.3,
    "一般般": 1.2,
    "下次别讲了": 1.7,
    "稍微有点少": 1.3,
    "有点少": 1.2,
    "不足": 1.1,
    "不清楚": 1.7,
    "听不懂": 1.8,
    "枯燥": 1.5,
    "无聊": 1.5,
    "太快": 1.3,
    "太难": 1.5,
    "有点难": 1.1,
    "压力": 1.3,
    "作业量有点大": 1.6,
    "作业太多": 1.8,
    "负担": 1.4,
    "混乱": 1.6,
    "失望": 1.7,
    "不满意": 1.7,
    "建议改进": 0.9,
}

NEGATIONS = ("不", "没", "未", "并非", "不是")

ASPECTS = {
    "课程内容": ("内容", "概念", "知识", "公式", "案例", "重点", "理论"),
    "授课方式": ("老师", "教师", "讲解", "讲课", "讲得", "语速", "板书", "演示", "深入浅出"),
    "课堂互动": ("互动", "讨论", "提问", "课堂", "参与", "交流", "回答"),
    "学习负担": ("作业", "任务", "难度", "压力", "负担", "考试", "练习"),
    "资源支持": ("课件", "资料", "视频", "平台", "教材", "答疑", "反馈"),
    "实验实践": ("实验", "实践", "实训", "动手", "上机"),
}


@dataclass
class TextSignal:
    score: float
    quality: float
    evidence: list[TextEvidence]
    aspects: list[AspectResult]


@dataclass
class ImageSignal:
    score: float
    quality: float
    evidence: list[ImageEvidence]
    description: str


def _sentiment(score: float, threshold: float = 0.14) -> str:
    if score > threshold:
        return "positive"
    if score < -threshold:
        return "negative"
    return "neutral"


def _bounded(value: float) -> int:
    return max(0, min(100, round(value)))


def _find_aspect(text: str, start: int) -> str:
    nearest: tuple[int, str] | None = None
    for name, words in ASPECTS.items():
        for word in words:
            for match in re.finditer(re.escape(word), text):
                distance = min(abs(start - match.start()), abs(start - match.end()))
                if distance <= 16 and (nearest is None or distance < nearest[0]):
                    nearest = (distance, name)
    return nearest[1] if nearest else "整体体验"


def analyze_text(text: str) -> TextSignal:
    normalized = re.sub(r"\s+", " ", text.strip())
    evidence: list[TextEvidence] = []
    weighted_score = 0.0
    total_weight = 0.0

    candidates: list[tuple[int, str, float, str]] = []
    for term, weight in POSITIVE_TERMS.items():
        for match in re.finditer(re.escape(term), normalized):
            prefix = normalized[max(0, match.start() - 2) : match.start()]
            polarity = "negative" if any(prefix.endswith(neg) for neg in NEGATIONS) else "positive"
            candidates.append((match.start(), term, weight, polarity))
    for term, weight in NEGATIVE_TERMS.items():
        for match in re.finditer(re.escape(term), normalized):
            prefix = normalized[max(0, match.start() - 5) : match.start()]
            polarity = "positive" if any(neg in prefix for neg in NEGATIONS) else "negative"
            candidates.append((match.start(), term, weight, polarity))

    occupied: list[tuple[int, int]] = []
    for start, term, weight, polarity in sorted(candidates, key=lambda item: (-len(item[1]), item[0])):
        end = start + len(term)
        if any(not (end <= a or start >= b) for a, b in occupied):
            continue
        occupied.append((start, end))
        sign = 1 if polarity == "positive" else -1
        weighted_score += sign * weight
        total_weight += weight
        evidence.append(
            TextEvidence(
                text=term,
                score=_bounded(58 + weight * 19),
                polarity=polarity,
                aspect=_find_aspect(normalized, start),
            )
        )

    exclamation = min(0.35, normalized.count("！") * 0.08 + normalized.count("!") * 0.08)
    if total_weight:
        score = math.tanh(weighted_score / max(1.6, total_weight * 0.72))
        score = max(-1.0, min(1.0, score + math.copysign(exclamation, score)))
    else:
        score = 0.0

    aspect_results: list[AspectResult] = []
    for aspect_name, keywords in ASPECTS.items():
        mentioned = any(keyword in normalized for keyword in keywords)
        aspect_evidence = [item for item in evidence if item.aspect == aspect_name]
        if not mentioned and not aspect_evidence:
            continue
        aspect_raw = sum((1 if item.polarity == "positive" else -1) * item.score for item in aspect_evidence)
        denom = sum(item.score for item in aspect_evidence) or 1
        aspect_score = aspect_raw / denom
        aspect_results.append(
            AspectResult(
                name=aspect_name,
                sentiment=_sentiment(aspect_score),
                intensity=_bounded(45 + abs(aspect_score) * 48),
                evidence=[item.text for item in aspect_evidence] or ["检测到相关方面，但情感证据较弱"],
            )
        )

    if not aspect_results:
        aspect_results.append(
            AspectResult(
                name="整体体验",
                sentiment=_sentiment(score),
                intensity=_bounded(38 + abs(score) * 48),
                evidence=[item.text for item in evidence] or ["未检测到明确方面词"],
            )
        )

    length_quality = min(1.0, len(normalized) / 80)
    evidence_quality = min(1.0, len(evidence) / 4)
    quality = 0.32 + 0.33 * length_quality + 0.35 * evidence_quality
    return TextSignal(score=score, quality=quality, evidence=evidence[:10], aspects=aspect_results)


def analyze_image(image_bytes: bytes) -> ImageSignal:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image.thumbnail((960, 960))
    width, height = image.size
    hsv = image.convert("HSV")
    hsv_stat = ImageStat.Stat(hsv)
    brightness = hsv_stat.mean[2] / 255
    saturation = hsv_stat.mean[1] / 255
    variance = sum(ImageStat.Stat(image).var) / 3 / (255**2)

    # Visual sentiment is deliberately conservative: image aesthetics are not treated as student emotion.
    score = max(-0.35, min(0.35, (brightness - 0.5) * 0.38 + (saturation - 0.35) * 0.18))
    quality = max(0.25, min(0.72, 0.30 + variance * 2.4 + min(width * height / 1_000_000, 0.18)))

    evidence: list[ImageEvidence] = []
    cols, rows = 4, 4
    regions: list[tuple[float, int, int, float, float]] = []
    for row in range(rows):
        for col in range(cols):
            left = col * width // cols
            top = row * height // rows
            right = (col + 1) * width // cols
            bottom = (row + 1) * height // rows
            crop = image.crop((left, top, right, bottom)).convert("HSV")
            stat = ImageStat.Stat(crop)
            local_brightness = stat.mean[2] / 255
            local_saturation = stat.mean[1] / 255
            salience = abs(local_brightness - brightness) + abs(local_saturation - saturation) + sum(stat.var) / 3 / (255**2)
            regions.append((salience, row, col, local_brightness, local_saturation))

    for salience, row, col, local_brightness, local_saturation in sorted(regions, reverse=True)[:4]:
        vertical = ("上", "中上", "中下", "下")[row]
        horizontal = ("左", "中左", "中右", "右")[col]
        observation = (
            ("明亮" if local_brightness > 0.64 else "偏暗" if local_brightness < 0.36 else "明暗适中")
            + "、"
            + ("色彩鲜明" if local_saturation > 0.48 else "色彩克制")
        )
        evidence.append(
            ImageEvidence(region=f"{vertical}{horizontal}区域", score=_bounded(48 + salience * 150), observation=observation)
        )

    description = (
        f"图像尺寸为 {width}×{height}；整体亮度约 {_bounded(brightness * 100)}%，"
        f"色彩饱和度约 {_bounded(saturation * 100)}%。视觉结果仅描述可观察氛围，不用于评价个人。"
    )
    return ImageSignal(score=score, quality=quality, evidence=evidence, description=description)


def _emotion_scores(score: float) -> list[EmotionScore]:
    positive = max(0.0, score)
    negative = max(0.0, -score)
    neutral = 1 - min(1.0, abs(score))
    return [
        EmotionScore(type="满意/愉悦", score=_bounded(12 + positive * 78)),
        EmotionScore(type="困惑/担忧", score=_bounded(10 + negative * 66)),
        EmotionScore(type="压力/不满", score=_bounded(8 + negative * 76)),
        EmotionScore(type="平静/中性", score=_bounded(neutral * 82)),
    ]


def _recommendation(aspects: list[AspectResult]) -> str:
    negative = [item.name for item in aspects if item.sentiment == "negative"]
    positive = [item.name for item in aspects if item.sentiment == "positive"]
    suggestions = {
        "课程内容": "补充核心概念示例，并在章节末提供知识结构小结",
        "授课方式": "适当放慢关键步骤讲解，增加演示与即时确认",
        "课堂互动": "保留有效讨论，并增加低门槛参与方式",
        "学习负担": "将任务划分为基础与拓展层，标注预计完成时间",
        "资源支持": "完善课件索引、示例资料和课后答疑入口",
        "实验实践": "增加实验与实践内容，并提供可复现的操作步骤和示例数据",
        "整体体验": "结合原始反馈进行人工复核后再制定调整措施",
    }
    if negative:
        action = "；".join(suggestions.get(item, suggestions["整体体验"]) for item in negative[:2])
        prefix = f"优先关注{'、'.join(negative[:2])}："
    else:
        action = "持续收集阶段性匿名反馈，并对低置信度样本进行人工复核"
        prefix = f"保持{'、'.join(positive[:2]) or '当前教学安排'}的有效做法；"
    return prefix + action + "。"


def integrate_visual_semantics(
    result: AnalysisResponse,
    sentiment: str,
    intensity: int,
    confidence: int,
    reason: str,
) -> AnalysisResponse:
    """Fuse Qwen's semantic vision signal with traceable AEF-R evidence.

    Low-level image statistics remain useful quality evidence, but they cannot
    understand OCR slogans, scene meaning, or communicative intent. This second
    gate adds that semantic signal without mixing requests across modalities.
    """
    if sentiment not in {"positive", "neutral", "negative"}:
        return result

    bounded_intensity = _bounded(intensity)
    bounded_confidence = _bounded(confidence)
    direction = 1 if sentiment == "positive" else -1 if sentiment == "negative" else 0
    visual_score = direction * bounded_intensity / 100
    visual_quality = max(0.2, bounded_confidence / 100)
    visual_aspect = AspectResult(
        name="视觉表达",
        sentiment=sentiment,
        intensity=bounded_intensity,
        evidence=[reason[:180]],
    )

    if result.mode == "image":
        result.overall = sentiment
        result.intensity = bounded_intensity
        result.confidence = _bounded(result.confidence * 0.25 + bounded_confidence * 0.75)
        result.emotions = _emotion_scores(visual_score)
        result.aspects = [visual_aspect]
        result.explanation += f"Qwen视觉语义判断：{reason[:220]}"
        return result

    if result.mode != "multimodal":
        return result

    text_direction = 1 if result.overall == "positive" else -1 if result.overall == "negative" else 0
    text_score = text_direction * result.intensity / 100
    text_quality = max(0.35, min(0.95, result.confidence / 100))
    weight_total = text_quality + visual_quality
    text_weight = text_quality / weight_total
    visual_weight = visual_quality / weight_total
    combined = text_score * text_weight + visual_score * visual_weight

    text_sentiment = _sentiment(text_score)
    if text_sentiment == sentiment:
        level = "consistent"
        consistency_score = _bounded(82 + (1 - abs(text_score - visual_score)) * 16)
        explanation = "文本评价与图片传达的语义情感方向一致。"
        confidence_factor = 1.0
    elif text_sentiment == "neutral" or sentiment == "neutral":
        level = "partially_consistent"
        consistency_score = _bounded(68 - abs(text_score - visual_score) * 24)
        explanation = "一个模态的情感表达较弱，系统保留两侧证据并动态加权。"
        confidence_factor = 0.9
    else:
        level = "conflicting"
        consistency_score = _bounded(44 - (abs(text_score) + abs(visual_score)) * 20)
        explanation = "文本评价与图片语义呈现相反的情感线索，建议人工复核。"
        confidence_factor = 0.78

    result.overall = _sentiment(combined)
    result.intensity = _bounded(42 + abs(combined) * 55)
    result.confidence = _bounded(
        (text_quality * text_weight + visual_quality * visual_weight) * confidence_factor * 100
    )
    result.emotions = _emotion_scores(combined)
    result.consistency = ConsistencyResult(
        level=level,
        score=consistency_score,
        explanation=explanation,
    )
    result.dominant_source = (
        "text" if text_weight > 0.58 else "image" if visual_weight > 0.58 else "balanced"
    )
    result.aspects = [item for item in result.aspects if item.name != "视觉氛围"] + [visual_aspect]
    result.explanation += f"Qwen视觉语义判断：{reason[:220]}"
    return result


def integrate_text_semantics(
    result: AnalysisResponse,
    sentiment: str,
    intensity: int,
    confidence: int,
    reason: str,
    semantic_evidence: list[TextEvidence],
) -> AnalysisResponse:
    """Fuse lexicon evidence with Qwen semantics for short and colloquial text."""
    if result.mode not in {"text", "multimodal"} or sentiment not in {
        "positive",
        "neutral",
        "negative",
    }:
        return result

    rule_evidence = list(result.text_evidence)
    existing = {(item.text, item.polarity, item.aspect) for item in rule_evidence}
    for item in semantic_evidence:
        key = (item.text, item.polarity, item.aspect)
        redundant = any(
            previous.text in item.text
            and previous.polarity == item.polarity
            and previous.aspect == item.aspect
            for previous in result.text_evidence
        )
        if key not in existing and not redundant:
            result.text_evidence.append(item)
            existing.add(key)

    rule_total = sum(item.score for item in rule_evidence)
    rule_signed = sum(
        (1 if item.polarity == "positive" else -1 if item.polarity == "negative" else 0) * item.score
        for item in rule_evidence
    )
    rule_score = rule_signed / rule_total if rule_total else 0.0
    semantic_direction = 1 if sentiment == "positive" else -1 if sentiment == "negative" else 0
    semantic_score = semantic_direction * _bounded(intensity) / 100
    semantic_quality = max(0.25, _bounded(confidence) / 100)

    if rule_evidence:
        rule_quality = min(0.86, 0.28 + len(rule_evidence) * 0.11)
        rule_weight = rule_quality * 0.42
        semantic_weight = semantic_quality * 0.58
        combined = (rule_score * rule_weight + semantic_score * semantic_weight) / (
            rule_weight + semantic_weight
        )
        combined_confidence = rule_quality * 0.35 + semantic_quality * 0.65
    else:
        combined = semantic_score
        combined_confidence = semantic_quality * 0.92

    result.overall = _sentiment(combined)
    result.intensity = _bounded(min(95, max(35, abs(combined) * 100)))
    result.confidence = _bounded(min(0.96, combined_confidence) * 100)
    result.emotions = _emotion_scores(combined)

    grouped: dict[str, list[TextEvidence]] = {}
    for item in result.text_evidence:
        grouped.setdefault(item.aspect, []).append(item)
    if grouped:
        result.aspects = []
        for aspect, items in grouped.items():
            signed = sum(
                (1 if item.polarity == "positive" else -1 if item.polarity == "negative" else 0)
                * item.score
                for item in items
            )
            total = sum(item.score for item in items) or 1
            aspect_score = signed / total
            result.aspects.append(
                AspectResult(
                    name=aspect,
                    sentiment=_sentiment(aspect_score),
                    intensity=_bounded(45 + abs(aspect_score) * 48),
                    evidence=[item.text for item in items],
                )
            )
    elif result.aspects:
        result.aspects[0] = AspectResult(
            name=result.aspects[0].name,
            sentiment=sentiment,
            intensity=_bounded(intensity),
            evidence=[reason[:180]],
        )

    result.explanation += f"Qwen文本语义判断：{reason[:220]}"
    return result


def analyze(mode: str, text: str, image_bytes: bytes | None) -> AnalysisResponse:
    started = time.perf_counter()
    text_signal = analyze_text(text) if mode != "image" else None
    image_signal = analyze_image(image_bytes) if mode != "text" and image_bytes else None

    if mode == "text" and text_signal:
        combined = text_signal.score
        confidence = text_signal.quality
        dominant = "text"
        consistency = ConsistencyResult(
            level="not_applicable", score=100, explanation="当前任务仅包含文本模态。"
        )
    elif mode == "image" and image_signal:
        combined = image_signal.score
        confidence = image_signal.quality * 0.78
        dominant = "image"
        consistency = ConsistencyResult(
            level="not_applicable", score=100, explanation="当前任务仅包含图像模态。"
        )
    elif text_signal and image_signal:
        text_weight = text_signal.quality / (text_signal.quality + image_signal.quality)
        image_weight = 1 - text_weight
        combined = text_signal.score * text_weight + image_signal.score * image_weight
        delta = abs(text_signal.score - image_signal.score)
        opposite = text_signal.score * image_signal.score < 0 and abs(text_signal.score) + abs(image_signal.score) > 0.32
        if opposite:
            level = "conflicting"
        elif delta < 0.18:
            level = "consistent"
        else:
            level = "partially_consistent"
        consistency_score = _bounded(100 - delta * 76)
        consistency = ConsistencyResult(
            level=level,
            score=consistency_score,
            explanation=(
                "图文呈现相反的情感线索，系统保留分模态判断并建议人工复核。"
                if level == "conflicting"
                else "图文线索基本一致。"
                if level == "consistent"
                else "图文关注重点或情感强度存在差异，综合结果按证据质量动态加权。"
            ),
        )
        confidence = (text_signal.quality * text_weight + image_signal.quality * image_weight) * (0.82 if opposite else 1)
        dominant = "text" if text_weight > 0.58 else "image" if image_weight > 0.58 else "balanced"
    else:
        raise ValueError("缺少可分析的输入")

    overall = _sentiment(combined)
    aspects = text_signal.aspects if text_signal else [
        AspectResult(
            name="视觉氛围",
            sentiment=overall,
            intensity=_bounded(38 + abs(combined) * 65),
            evidence=[item.observation for item in (image_signal.evidence if image_signal else [])[:2]],
        )
    ]
    text_evidence = text_signal.evidence if text_signal else []
    image_evidence = image_signal.evidence if image_signal else []
    evidence_count = len(text_evidence) + len(image_evidence)
    insights = [
        f"识别到 {len(aspects)} 个课程评价方面和 {evidence_count} 条可回溯证据。",
        consistency.explanation,
    ]
    if confidence < 0.48:
        insights.append("当前输入证据较弱，建议结合更多匿名反馈后再作判断。")
    explanation_parts = []
    if text_signal:
        explanation_parts.append(f"文本侧提取到 {len(text_evidence)} 条方面相关证据。")
    if image_signal:
        explanation_parts.append(image_signal.description)
    explanation_parts.append("综合结论由 AEF-R Lite 动态门控融合生成。")

    return AnalysisResponse(
        analysis_id=str(uuid.uuid4()),
        mode=mode,
        overall=overall,
        intensity=_bounded(42 + abs(combined) * 55),
        confidence=_bounded(confidence * 100),
        emotions=_emotion_scores(combined),
        aspects=aspects,
        text_evidence=text_evidence,
        image_evidence=image_evidence,
        consistency=consistency,
        dominant_source=dominant,
        insights=insights,
        recommendation=_recommendation(aspects),
        explanation="".join(explanation_parts),
        engine="AEF-R Lite / evidence-first fusion",
        latency_ms=round((time.perf_counter() - started) * 1000),
    )
