'use client';

import { useMemo, useRef, useState } from 'react';
import {
  Activity,
  AlertCircle,
  BrainCircuit,
  CheckCircle2,
  ChevronRight,
  FileText,
  Image as ImageIcon,
  Layers3,
  Loader2,
  ScanSearch,
  ShieldCheck,
  Sparkles,
  UploadCloud,
} from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Textarea } from '@/components/ui/textarea';

type Mode = 'text' | 'image' | 'multimodal';
type Sentiment = 'positive' | 'neutral' | 'negative';

interface AnalysisResult {
  analysis_id: string;
  mode: Mode;
  overall: Sentiment;
  intensity: number;
  confidence: number;
  emotions: Array<{ type: string; score: number }>;
  aspects: Array<{ name: string; sentiment: Sentiment; intensity: number; evidence: string[] }>;
  text_evidence: Array<{ text: string; score: number; polarity: Sentiment; aspect: string }>;
  image_evidence: Array<{ region: string; score: number; observation: string }>;
  consistency: {
    level: 'consistent' | 'partially_consistent' | 'conflicting' | 'not_applicable';
    score: number;
    explanation: string;
  };
  dominant_source: 'text' | 'image' | 'balanced' | 'not_applicable';
  insights: string[];
  recommendation: string;
  explanation: string;
  engine: string;
  latency_ms: number;
}

const modes: Array<{
  id: Mode;
  label: string;
  hint: string;
  icon: typeof FileText;
}> = [
  { id: 'text', label: '文本分析', hint: '识别观点与细粒度情绪', icon: FileText },
  { id: 'image', label: '图片分析', hint: '定位视觉情感证据', icon: ImageIcon },
  { id: 'multimodal', label: '图文融合', hint: '判断跨模态一致性与冲突', icon: Layers3 },
];

const defaultText =
  '《机器学习导论》真的太棒了！老师讲得深入浅出，能把复杂的公式讲得很清楚。课程作业虽然有挑战性，但确实能帮助我们巩固知识。唯一的不足是实验部分稍微有点少，希望以后能增加一些实践内容。';

const sentimentLabels: Record<Sentiment, string> = {
  positive: '积极',
  neutral: '中性',
  negative: '消极',
};

const consistencyLabels = {
  consistent: '一致',
  partially_consistent: '部分一致',
  conflicting: '存在冲突',
  not_applicable: '单模态',
};

export default function Home() {
  const [mode, setMode] = useState<Mode>('text');
  const [text, setText] = useState(defaultText);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState('');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);
  const requestRef = useRef<AbortController | null>(null);

  const readiness = useMemo(() => {
    if (mode === 'text') return text.trim() ? 100 : 0;
    if (mode === 'image') return imageFile ? 100 : 0;
    return (text.trim() ? 50 : 0) + (imageFile ? 50 : 0);
  }, [imageFile, mode, text]);

  const invalidateAnalysis = () => {
    requestRef.current?.abort();
    requestRef.current = null;
    setIsAnalyzing(false);
    setResult(null);
    setError('');
  };

  const selectMode = (nextMode: Mode) => {
    invalidateAnalysis();
    setMode(nextMode);
  };

  const analyzeFeedback = async () => {
    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;
    setIsAnalyzing(true);
    setError('');
    const formData = new FormData();
    formData.append('mode', mode);
    if (mode !== 'image') formData.append('text', text);
    if (mode !== 'text' && imageFile) formData.append('image', imageFile);
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/analyze`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
        cache: 'no-store',
      });
      const data: unknown = await response.json();
      if (!response.ok) {
        const detail =
          typeof data === 'object' && data !== null && 'detail' in data && typeof data.detail === 'string'
            ? data.detail
            : '分析服务返回异常';
        throw new Error(detail);
      }
      if (requestRef.current !== controller) return;
      if (typeof data !== 'object' || data === null || !('mode' in data) || data.mode !== mode) {
        throw new Error('分析结果与当前模式不一致，已阻止显示。');
      }
      setResult(data as AnalysisResult);
    } catch (reason) {
      if (reason instanceof DOMException && reason.name === 'AbortError') return;
      setError(reason instanceof Error ? reason.message : '无法连接分析服务，请确认后端已启动。');
    } finally {
      if (requestRef.current === controller) {
        requestRef.current = null;
        setIsAnalyzing(false);
      }
    }
  };

  const loadExample = () => {
    invalidateAnalysis();
    setMode('text');
    setText(defaultText);
  };

  return (
    <main className="min-h-screen bg-background text-foreground">
      <header className="sticky top-0 z-20 border-b border-border/70 bg-background/88 backdrop-blur-xl">
        <div className="mx-auto flex h-16 max-w-[1480px] items-center justify-between px-4 md:px-7">
          <div className="flex items-center gap-3">
            <div className="grid size-10 place-items-center rounded-xl bg-primary text-primary-foreground shadow-[0_8px_24px_rgba(12,101,91,.22)]">
              <BrainCircuit className="size-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-heading text-base font-semibold tracking-tight">教情智析</span>
                <Badge variant="secondary" className="hidden text-[10px] sm:inline-flex">
                  Qwen3.5 9B
                </Badge>
              </div>
              <p className="text-xs text-muted-foreground">证据引导的多模态课程反馈分析</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <div className="hidden items-center gap-2 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-xs font-medium text-emerald-800 md:flex">
              <span className="size-1.5 rounded-full bg-emerald-500" />
              本地隐私模式
            </div>
            <Button variant="outline" size="sm" onClick={loadExample}>
              载入示例
            </Button>
          </div>
        </div>
      </header>

      <div className="mx-auto grid max-w-[1480px] gap-6 px-4 py-6 lg:grid-cols-[260px_minmax(0,1fr)] lg:px-7">
        <aside className="space-y-5">
          <div className="rounded-2xl border border-border/80 bg-card p-3 shadow-sm">
            <p className="px-3 pb-2 pt-1 text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
              分析工作台
            </p>
            <nav className="space-y-1" aria-label="分析模式">
              {modes.map((item) => {
                const Icon = item.icon;
                const selected = mode === item.id;
                return (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => selectMode(item.id)}
                    className={`group flex w-full items-center gap-3 rounded-xl px-3 py-3 text-left transition ${
                      selected
                        ? 'bg-primary text-primary-foreground shadow-sm'
                        : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                    }`}
                  >
                    <Icon className="size-4 shrink-0" />
                    <span className="min-w-0 flex-1">
                      <span className="block text-sm font-semibold">{item.label}</span>
                      <span className={`block truncate text-[11px] ${selected ? 'text-primary-foreground/70' : ''}`}>
                        {item.hint}
                      </span>
                    </span>
                    <ChevronRight className={`size-4 transition ${selected ? 'opacity-100' : 'opacity-0 group-hover:opacity-50'}`} />
                  </button>
                );
              })}
            </nav>
          </div>

          <Card className="border-0 bg-[#123c3a] text-white shadow-lg">
            <CardHeader>
              <div className="mb-1 grid size-9 place-items-center rounded-lg bg-white/10">
                <ShieldCheck className="size-4 text-[#8fe0cf]" />
              </div>
              <CardTitle className="text-white">可信分析原则</CardTitle>
              <CardDescription className="text-white/62">
                不依据单张图片评价个人；证据不足时明确提示人工复核。
              </CardDescription>
            </CardHeader>
          </Card>
        </aside>

        <section className="min-w-0 space-y-6">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <div className="mb-2 flex items-center gap-2 text-xs font-semibold text-primary">
                <Sparkles className="size-3.5" />
                多模态证据工作流
              </div>
              <h1 className="font-heading text-2xl font-semibold tracking-tight md:text-3xl">
                先定位证据，再理解情感
              </h1>
              <p className="mt-1 max-w-2xl text-sm leading-6 text-muted-foreground">
                围绕课程内容、授课方式、课堂互动与学习负担，分别分析图文线索并保留模态冲突。
              </p>
            </div>
            <Badge variant="outline" className="w-fit gap-1.5 py-1.5">
              <Activity className="size-3" /> AEF-R × Qwen3.5 本地融合
            </Badge>
          </div>

          <div className="grid gap-6 xl:grid-cols-[minmax(0,1.08fr)_minmax(380px,.92fr)]">
            <Card className="shadow-[0_18px_50px_rgba(31,54,51,.08)]">
              <CardHeader className="border-b border-border/70">
                <CardTitle>输入课程反馈</CardTitle>
                <CardDescription>当前模式：{modes.find((item) => item.id === mode)?.label}</CardDescription>
              </CardHeader>
              <CardContent className="space-y-5 pt-1">
                {mode !== 'image' && (
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <label htmlFor="feedback" className="text-sm font-semibold">
                        文字反馈
                      </label>
                      <span className="text-xs text-muted-foreground">{text.length} / 1000</span>
                    </div>
                    <Textarea
                      id="feedback"
                      value={text}
                      onChange={(event) => {
                        invalidateAnalysis();
                        setText(event.target.value.slice(0, 1000));
                      }}
                      className="min-h-36 resize-none bg-muted/35 text-[15px] leading-7"
                      placeholder="粘贴匿名课程评价、问卷开放题或课堂留言……"
                    />
                  </div>
                )}

                {mode !== 'text' && (
                  <div className="space-y-2">
                    <span className="text-sm font-semibold">课堂图片</span>
                    <button
                      type="button"
                      onClick={() => inputRef.current?.click()}
                      className="flex w-full items-center gap-4 rounded-xl border border-dashed border-primary/35 bg-primary/[.035] p-4 text-left transition hover:border-primary/60 hover:bg-primary/[.06]"
                    >
                      <span className="grid size-11 shrink-0 place-items-center rounded-xl bg-primary/10 text-primary">
                        <UploadCloud className="size-5" />
                      </span>
                      <span className="min-w-0 flex-1">
                        <span className="block truncate text-sm font-semibold">{imageFile?.name || '点击上传 JPG、PNG 或 WebP'}</span>
                        <span className="mt-0.5 block text-xs text-muted-foreground">仅提取与教学反馈相关的视觉证据</span>
                      </span>
                      {imageFile && <CheckCircle2 className="size-5 text-emerald-600" />}
                    </button>
                    <input
                      ref={inputRef}
                      className="hidden"
                      type="file"
                      accept="image/jpeg,image/png,image/webp"
                      onChange={(event) => {
                        invalidateAnalysis();
                        const file = event.target.files?.[0] ?? null;
                        setImageFile(file);
                        setImagePreview((previous) => {
                          if (previous) URL.revokeObjectURL(previous);
                          return file ? URL.createObjectURL(file) : '';
                        });
                      }}
                    />
                    {imagePreview && (
                      // oxlint-disable-next-line next/no-img-element
                      <img
                        src={imagePreview}
                        alt="待分析课堂图片预览"
                        className="max-h-44 w-full rounded-xl border border-border/70 object-contain"
                      />
                    )}
                  </div>
                )}

                <div className="rounded-xl bg-muted/55 p-4">
                  <div className="mb-2 flex items-center justify-between text-xs">
                    <span className="font-medium">输入完整度</span>
                    <span className="font-semibold text-primary">{readiness}%</span>
                  </div>
                  <Progress value={readiness} />
                </div>

                {error && (
                  <div role="alert" className="flex gap-2 rounded-xl border border-rose-200 bg-rose-50 p-3 text-sm text-rose-800">
                    <AlertCircle className="mt-0.5 size-4 shrink-0" />
                    <span>{error}</span>
                  </div>
                )}

                <Button className="h-11 w-full gap-2 text-sm" disabled={readiness < 100 || isAnalyzing} onClick={analyzeFeedback}>
                  {isAnalyzing ? <Loader2 className="size-4 animate-spin" /> : <ScanSearch className="size-4" />}
                  {isAnalyzing ? '正在定位跨模态证据…' : '开始证据增强分析'}
                </Button>
              </CardContent>
            </Card>

            <Card className="border-primary/15 bg-[linear-gradient(155deg,rgba(255,255,255,.98),rgba(237,247,244,.88))] shadow-[0_18px_50px_rgba(31,54,51,.08)]">
              <CardHeader className="border-b border-primary/10">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <CardTitle>证据预览</CardTitle>
                    <CardDescription>
                      {result
                        ? `本次结果：${modes.find((item) => item.id === result.mode)?.label}`
                        : '展示系统如何得出结论，而不只给出标签'}
                    </CardDescription>
                  </div>
                  {result && (
                    <Badge className="bg-amber-100 text-amber-900 hover:bg-amber-100">
                      {sentimentLabels[result.overall]}
                    </Badge>
                  )}
                </div>
              </CardHeader>
              <CardContent className="space-y-5 pt-1">
                {!result ? (
                  <div className="grid min-h-[480px] place-items-center text-center">
                    <div className="max-w-xs">
                      <span className="mx-auto grid size-14 place-items-center rounded-2xl bg-primary/10 text-primary">
                        <ScanSearch className="size-6" />
                      </span>
                      <h3 className="mt-4 font-semibold">等待分析</h3>
                      <p className="mt-1 text-sm leading-6 text-muted-foreground">
                        结果将展示方面级情感、文本证据、视觉区域、模态一致性与教学建议。
                      </p>
                    </div>
                  </div>
                ) : (
                  <>
                    <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                      {[
                        ['综合情感', sentimentLabels[result.overall], result.overall === 'positive' ? 'text-emerald-700' : result.overall === 'negative' ? 'text-rose-700' : 'text-slate-700'],
                        ['情绪强度', `${result.intensity}%`, 'text-amber-700'],
                        ['置信度', `${result.confidence}%`, 'text-primary'],
                        ['一致性', consistencyLabels[result.consistency.level], 'text-amber-700'],
                      ].map(([label, value, tone]) => (
                        <div key={label} className="rounded-xl border border-white/80 bg-white/72 p-3 shadow-sm">
                          <p className="text-[11px] text-muted-foreground">{label}</p>
                          <p className={`mt-1 text-base font-semibold ${tone}`}>{value}</p>
                        </div>
                      ))}
                    </div>

                    <div className="grid gap-3 lg:grid-cols-[1.2fr_.8fr]">
                      <div className="rounded-xl border border-border/70 bg-white/75 p-4">
                        <p className="mb-3 text-xs font-semibold uppercase tracking-[0.12em] text-muted-foreground">
                          语义洞察
                        </p>
                        <ul className="space-y-2 text-sm leading-6">
                          {result.insights.map((item, index) => (
                            <li key={`${item}-${index}`} className="flex gap-2">
                              <span className="mt-2 size-1.5 shrink-0 rounded-full bg-primary" />
                              <span>{item}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                      <div className="rounded-xl border border-border/70 bg-white/75 p-4">
                        <p className="mb-3 text-xs font-semibold uppercase tracking-[0.12em] text-muted-foreground">
                          情绪分布
                        </p>
                        <div className="space-y-3">
                          {result.emotions.map((emotion) => (
                            <div key={emotion.type}>
                              <div className="mb-1 flex justify-between text-xs">
                                <span>{emotion.type}</span>
                                <span className="font-medium">{emotion.score}%</span>
                              </div>
                              <Progress value={emotion.score} className="h-1.5" />
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>

                    {result.text_evidence.length > 0 && (
                      <div className="space-y-3">
                        <p className="text-xs font-semibold uppercase tracking-[0.12em] text-muted-foreground">关键文本证据</p>
                        <div className="flex flex-wrap gap-2 rounded-xl border border-border/70 bg-white/75 p-4">
                          {result.text_evidence.map((item, index) => (
                            <span
                              key={`${item.text}-${index}`}
                              className={`rounded-lg px-2.5 py-1.5 text-xs font-medium ${item.polarity === 'positive' ? 'bg-emerald-100 text-emerald-900' : 'bg-rose-100 text-rose-900'}`}
                              title={`${item.aspect} · 证据强度 ${item.score}%`}
                            >
                              {item.text} · {item.aspect}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {result.image_evidence.length > 0 && (
                      <div className="space-y-3">
                        <p className="text-xs font-semibold uppercase tracking-[0.12em] text-muted-foreground">视觉区域证据</p>
                        <div className="grid grid-cols-2 gap-2">
                          {result.image_evidence.map((item) => (
                            <div key={item.region} className="rounded-xl border border-border/70 bg-white/75 p-3 text-xs">
                              <p className="font-semibold">{item.region} · {item.score}%</p>
                              <p className="mt-1 text-muted-foreground">{item.observation}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    <div className="space-y-3">
                      <p className="text-xs font-semibold uppercase tracking-[0.12em] text-muted-foreground">方面级判断</p>
                      <div className="space-y-3">
                        {result.aspects.map((item) => (
                          <div key={item.name}>
                            <div className="mb-1.5 flex items-center justify-between text-xs">
                              <span className="font-medium">{item.name}</span>
                              <span className="text-muted-foreground">{sentimentLabels[item.sentiment]} · {item.intensity}%</span>
                            </div>
                            <Progress value={item.intensity} className="h-1.5" />
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="rounded-xl border border-[#d5c08b] bg-[#fff9e9] p-4">
                      <div className="mb-1.5 flex items-center gap-2 text-sm font-semibold text-[#6f5520]">
                        <Sparkles className="size-4" /> 教学改进建议
                      </div>
                      <p className="text-sm leading-6 text-[#76643e]">{result.recommendation}</p>
                      <p className="mt-3 border-t border-[#e9dbb7] pt-3 text-[11px] text-[#8c7a51]">
                        {result.engine} · {result.latency_ms} ms
                      </p>
                    </div>
                  </>
                )}
              </CardContent>
            </Card>
          </div>
        </section>
      </div>
    </main>
  );
}
