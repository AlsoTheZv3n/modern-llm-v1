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

public partial class MainViewModel : ObservableObject
{
    private const int MaxPoints = 5000;

    private JsonlMetricReader? _reader;

    public ObservableCollection<ObservablePoint> TrainLossPoints { get; } = new();
    public ObservableCollection<ObservablePoint> ValLossPoints { get; } = new();

    public ISeries[] LossSeries { get; }

    public Axis[] XAxes { get; } = new Axis[]
    {
        new Axis { Name = "step", NameTextSize = 11, TextSize = 10 }
    };

    public Axis[] YAxes { get; } = new Axis[]
    {
        new Axis { Name = "loss", NameTextSize = 11, TextSize = 10, MinLimit = 0 }
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
            }
        });
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
