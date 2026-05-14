using System;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

using ModernLLM.Monitor.Models;
using ModernLLM.Monitor.Services;

namespace ModernLLM.Monitor.ViewModels;

public partial class SamplerViewModel : ObservableObject
{
    private readonly string _repoRoot;
    private CancellationTokenSource? _cts;

    public ObservableCollection<CheckpointInfo> Checkpoints { get; } = new();

    [ObservableProperty]
    private CheckpointInfo? _selectedCheckpoint;

    [ObservableProperty]
    private string _metaPath = "";

    [ObservableProperty]
    private string _prompt = "The history of Rome";

    [ObservableProperty]
    private int _numTokens = 200;

    [ObservableProperty]
    private double _temperature = 0.7;

    [ObservableProperty]
    private int _seqLen = 512;

    [ObservableProperty]
    private int _dModel = 384;

    [ObservableProperty]
    private int _nHeads = 6;

    [ObservableProperty]
    private int _nKvHeads = 2;

    [ObservableProperty]
    private int _nLayers = 6;

    [ObservableProperty]
    private int _dFfn = 1536;

    [ObservableProperty]
    private int _seed = 42;

    [ObservableProperty]
    private string _output = "";

    [ObservableProperty]
    private string _stderrOutput = "";

    [ObservableProperty]
    private bool _isSampling;

    [ObservableProperty]
    private string _status = "ready";

    public SamplerViewModel(string repoRoot)
    {
        _repoRoot = repoRoot;
        RefreshCheckpoints();
        AutoFillMeta();
    }

    [RelayCommand]
    public void RefreshCheckpoints()
    {
        Checkpoints.Clear();
        var runs = CheckpointBrowser.FindRunsDir(_repoRoot, 2)
                    ?? Path.Combine(_repoRoot, "runs");
        foreach (var ck in CheckpointBrowser.Discover(runs))
            Checkpoints.Add(ck);
        if (SelectedCheckpoint == null && Checkpoints.Count > 0)
            SelectedCheckpoint = Checkpoints[0];
        Status = $"{Checkpoints.Count} checkpoint(s) in {runs}";
    }

    /// <summary>Auto-pick the largest .meta in data/ as a starting point.</summary>
    private void AutoFillMeta()
    {
        var dataDir = Path.Combine(_repoRoot, "data");
        if (!Directory.Exists(dataDir)) return;
        var newest = Directory.GetFiles(dataDir, "*.meta", SearchOption.TopDirectoryOnly)
                               .Select(p => new FileInfo(p))
                               .OrderByDescending(f => f.Length)
                               .FirstOrDefault();
        if (newest != null) MetaPath = newest.FullName;
    }

    [RelayCommand]
    public async Task SampleAsync()
    {
        if (SelectedCheckpoint == null)
        {
            Status = "no checkpoint selected";
            return;
        }
        if (string.IsNullOrWhiteSpace(MetaPath) || !File.Exists(MetaPath))
        {
            Status = $"meta file not found: {MetaPath}";
            return;
        }

        IsSampling = true;
        Status = "sampling…";
        Output = "";
        StderrOutput = "";

        _cts?.Cancel();
        _cts = new CancellationTokenSource();

        var cfg = new SampleConfig(
            CheckpointPath: SelectedCheckpoint.Path,
            MetaPath: MetaPath,
            Prompt: Prompt,
            NumTokens: NumTokens,
            SeqLen: SeqLen,
            DModel: DModel,
            NHeads: NHeads,
            NKvHeads: NKvHeads,
            NLayers: NLayers,
            DFfn: DFfn,
            Temperature: Temperature,
            Seed: Seed
        );

        var t0 = DateTime.UtcNow;
        try
        {
            var result = await SamplerService.RunAsync(_repoRoot, cfg, _cts.Token);
            var elapsed = (DateTime.UtcNow - t0).TotalSeconds;
            Output = result.Text;
            StderrOutput = result.Stderr;
            Status = result.Success
                ? $"done in {elapsed:F1}s"
                : $"failed (exit {result.ExitCode}) in {elapsed:F1}s";
        }
        finally
        {
            IsSampling = false;
        }
    }

    [RelayCommand]
    public void Cancel()
    {
        _cts?.Cancel();
        Status = "cancelling…";
    }
}
