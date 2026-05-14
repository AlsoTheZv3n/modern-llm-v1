using System;
using System.Collections.ObjectModel;
using System.Windows;

using CommunityToolkit.Mvvm.ComponentModel;

using LiveChartsCore;
using LiveChartsCore.Defaults;
using LiveChartsCore.SkiaSharpView;
using LiveChartsCore.SkiaSharpView.Painting;
using SkiaSharp;

using ModernLLM.Monitor.Models;
using ModernLLM.Monitor.Services;

namespace ModernLLM.Monitor.ViewModels;

public partial class MainViewModel : ObservableObject, IDisposable
{
    private const int MaxPoints = 5000;
    private const int ThroughputWindow = 10;  // rolling-window size for steps/sec

    private JsonlMetricReader? _reader;
    private readonly NvidiaSmiPoller _gpu = new NvidiaSmiPoller(2.0);

    // Recent (step, elapsed_sec) pairs to compute throughput from. Bounded.
    private readonly System.Collections.Generic.Queue<(int step, double elapsed)> _recent =
        new System.Collections.Generic.Queue<(int, double)>();

    public ObservableCollection<ObservablePoint> TrainLossPoints { get; } = new();
    public ObservableCollection<ObservablePoint> ValLossPoints { get; } = new();

    public ISeries[] LossSeries { get; }

    public Axis[] XAxes { get; } = new Axis[]
    {
        new Axis { Name = "step", NameTextSize = 11, TextSize = 10 }
    };

    public Axis[] YAxes { get; } = new Axis[]
    {
        // MinLimit toggled by AutoZoomY: null = auto-fit to data; 0 = anchored.
        new Axis { Name = "loss", NameTextSize = 11, TextSize = 10, MinLimit = null }
    };

    [ObservableProperty]
    private string _currentFile = "(no file)";

    [ObservableProperty]
    private int _step;

    [ObservableProperty]
    private double _loss;

    [ObservableProperty]
    private double _valLoss = double.NaN;

    [ObservableProperty]
    private double _lr;

    [ObservableProperty]
    private double _gradNorm;

    [ObservableProperty]
    private long _tokensSeen;

    [ObservableProperty]
    private double _elapsedSec;

    [ObservableProperty]
    private string _status = "Idle";

    // Progress tracking — TotalSteps is set by the user (toolbar input).
    // Defaults match the current Tier-3 50M run.
    [ObservableProperty]
    private int _totalSteps = 10000;

    [ObservableProperty]
    private long _totalTokens = 1_710_000_000;   // 1.8B * (1 - val_frac 0.05)

    [ObservableProperty]
    private double _progressPct;     // 0..100 (Step / TotalSteps)

    [ObservableProperty]
    private double _tokensPct;       // 0..100 (TokensSeen / TotalTokens)

    [ObservableProperty]
    private string _etaText = "ETA: —";

    [ObservableProperty]
    private string _throughputText = "—";

    // Last non-NaN val loss (Val gets logged only every val-every steps, so the
    // raw ValLoss is NaN most of the time — keep the most recent real value).
    [ObservableProperty]
    private double _lastValLoss = double.NaN;

    [ObservableProperty]
    private string _lastValAt = "—";

    [ObservableProperty]
    private bool _autoZoomY = true;  // start zoomed to recent so late training is visible

    // GPU stats — populated by NvidiaSmiPoller. GpuAvailable goes false on
    // first nvidia-smi failure (e.g. not installed / not on PATH).
    [ObservableProperty]
    private bool _gpuAvailable;

    [ObservableProperty]
    private int _gpuUtilPct;

    [ObservableProperty]
    private int _gpuVramUsedMB;

    [ObservableProperty]
    private int _gpuVramTotalMB;

    [ObservableProperty]
    private int _gpuTempC;

    [ObservableProperty]
    private double _gpuPowerW;

    [ObservableProperty]
    private string _gpuStatus = "GPU: querying nvidia-smi…";

    public MainViewModel()
    {
        LossSeries = new ISeries[]
        {
            new LineSeries<ObservablePoint>
            {
                Name = "train",
                Values = TrainLossPoints,
                Stroke = new SolidColorPaint(SKColors.CornflowerBlue, 2),
                Fill = null,
                GeometrySize = 0,
                LineSmoothness = 0.2,
            },
            new LineSeries<ObservablePoint>
            {
                Name = "val",
                Values = ValLossPoints,
                Stroke = new SolidColorPaint(SKColors.OrangeRed, 2),
                Fill = null,
                GeometrySize = 6,
                GeometryStroke = new SolidColorPaint(SKColors.OrangeRed, 2),
                GeometryFill = new SolidColorPaint(SKColors.OrangeRed),
                LineSmoothness = 0,
            }
        };

        _gpu.OnSample += HandleGpuSample;
        _gpu.OnError += HandleGpuError;
        _gpu.Start();
    }

    private void HandleGpuSample(GpuStats s)
    {
        Application.Current?.Dispatcher.Invoke(() =>
        {
            GpuAvailable = true;
            GpuUtilPct = s.UtilPct;
            GpuVramUsedMB = s.VramUsedMB;
            GpuVramTotalMB = s.VramTotalMB;
            GpuTempC = s.TempC;
            GpuPowerW = s.PowerW;
            GpuStatus = $"GPU: {s.UtilPct}% / {s.VramUsedMB:N0}/{s.VramTotalMB:N0} MB / {s.TempC}°C / {s.PowerW:F0} W";
        });
    }

    private void HandleGpuError(string msg)
    {
        Application.Current?.Dispatcher.Invoke(() =>
        {
            GpuAvailable = false;
            GpuStatus = $"GPU: {msg}";
        });
    }

    public void Dispose()
    {
        _gpu.Dispose();
        _reader?.Dispose();
    }

    public void OpenFile(string path)
    {
        _reader?.Dispose();
        _reader = new JsonlMetricReader(path);
        _reader.OnSample += HandleSample;
        _reader.OnReset += HandleReset;
        _reader.Start();
        CurrentFile = path;
        Status = $"Tailing {System.IO.Path.GetFileName(path)}";
    }

    public void CloseFile()
    {
        _reader?.Dispose();
        _reader = null;
        Status = "Idle";
        CurrentFile = "(no file)";
    }

    private void HandleSample(MetricSample s)
    {
        Application.Current.Dispatcher.Invoke(() =>
        {
            Step = s.Step;
            Loss = s.Loss;
            ValLoss = s.ValLoss;
            Lr = s.Lr;
            GradNorm = s.GradNorm;
            TokensSeen = s.TokensSeen;
            ElapsedSec = s.ElapsedSec;

            if (!double.IsNaN(s.Loss))
            {
                TrainLossPoints.Add(new ObservablePoint(s.Step, s.Loss));
                while (TrainLossPoints.Count > MaxPoints)
                    TrainLossPoints.RemoveAt(0);
            }
            if (!double.IsNaN(s.ValLoss))
            {
                ValLossPoints.Add(new ObservablePoint(s.Step, s.ValLoss));
                while (ValLossPoints.Count > MaxPoints)
                    ValLossPoints.RemoveAt(0);

                LastValLoss = s.ValLoss;
                LastValAt = $"@ step {s.Step}";
            }

            // Throughput from rolling (step, elapsed) window. Gradient avoids
            // skewing by an outlier startup time.
            _recent.Enqueue((s.Step, s.ElapsedSec));
            while (_recent.Count > ThroughputWindow) _recent.Dequeue();

            UpdateProgressAndEta();
        });
    }

    private void UpdateProgressAndEta()
    {
        ProgressPct = TotalSteps > 0
            ? Math.Min(100.0, 100.0 * Step / TotalSteps)
            : 0;
        TokensPct = TotalTokens > 0
            ? Math.Min(100.0, 100.0 * TokensSeen / TotalTokens)
            : 0;

        if (_recent.Count < 2)
        {
            ThroughputText = "warming up…";
            EtaText = "ETA: —";
            return;
        }

        var first = _recent.Peek();
        var last = _recent.ToArray()[^1];
        double dSteps = last.step - first.step;
        double dElapsed = last.elapsed - first.elapsed;
        if (dSteps <= 0 || dElapsed <= 0)
        {
            ThroughputText = "—";
            EtaText = "ETA: —";
            return;
        }

        double secPerStep = dElapsed / dSteps;
        double tokPerSec = (double)(TokensSeen) / Math.Max(ElapsedSec, 1e-6);

        ThroughputText = $"{secPerStep,4:F2} sec/step   ·   {tokPerSec / 1000.0,5:F1}k tok/s";

        int remaining = Math.Max(0, TotalSteps - Step);
        double etaSec = remaining * secPerStep;
        EtaText = $"ETA: {FormatDuration(etaSec)}";
    }

    private static string FormatDuration(double sec)
    {
        if (sec <= 0 || double.IsNaN(sec) || double.IsInfinity(sec)) return "—";
        var ts = TimeSpan.FromSeconds(sec);
        if (ts.TotalDays >= 1) return $"{(int)ts.TotalDays}d {ts.Hours}h {ts.Minutes}m";
        if (ts.TotalHours >= 1) return $"{(int)ts.TotalHours}h {ts.Minutes}m";
        if (ts.TotalMinutes >= 1) return $"{(int)ts.TotalMinutes}m {ts.Seconds}s";
        return $"{ts.Seconds}s";
    }

    // Recompute progress when user edits the totals.
    partial void OnTotalStepsChanged(int value) => UpdateProgressAndEta();
    partial void OnTotalTokensChanged(long value) => UpdateProgressAndEta();

    // Auto-zoom Y axis: when on, LiveCharts auto-fits to the visible data
    // (so late-training fluctuations of ~0.1 are visible); when off, anchor
    // to 0 to show the full history including the initial loss=11.6.
    partial void OnAutoZoomYChanged(bool value)
    {
        YAxes[0].MinLimit = value ? null : (double?)0;
    }

    private void HandleReset()
    {
        Application.Current.Dispatcher.Invoke(() =>
        {
            TrainLossPoints.Clear();
            ValLossPoints.Clear();
            Status = $"Reset (file truncated): {CurrentFile}";
        });
    }
}
