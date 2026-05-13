# 05 — C# Monitoring GUI

> Real-time training monitor, checkpoint manager, and training controller.
> Built in C# (.NET 8, WPF) communicating with the C++ trainer via named pipes.

---

## Architecture

```
C++ Trainer Process
    ↓ (Named pipe / Unix socket)
C# Monitor Process
    ↓
WPF GUI Window
```

The C++ trainer emits JSON metrics every N steps. The C# GUI reads them and renders live charts, stats, and controls.

---

## Communication Protocol

### C++ side — emit metrics every step

```cpp
struct TrainingMetrics {
    int    step;
    float  loss;
    float  lr;
    float  grad_norm;
    float  tokens_per_sec;
    float  mfu;              // Model FLOP utilization (%)
    float  gpu_memory_gb;
    float  aux_loss;         // MoE load balancing loss
    long   tokens_seen;
    float  eval_perplexity;  // set only when eval runs
};

void emit_metrics(const TrainingMetrics& m, HANDLE pipe) {
    char buf[512];
    snprintf(buf, sizeof(buf),
        "{\"step\":%d,\"loss\":%.4f,\"lr\":%.2e,"
        "\"grad_norm\":%.4f,\"tok_per_sec\":%.0f,"
        "\"mfu\":%.1f,\"tokens_seen\":%ld,"
        "\"aux_loss\":%.4f}\n",
        m.step, m.loss, m.lr, m.grad_norm,
        m.tokens_per_sec, m.mfu, m.tokens_seen, m.aux_loss);
    WriteFile(pipe, buf, strlen(buf), nullptr, nullptr);
}
```

### C# side — read loop

```csharp
// MetricsReader.cs
public class MetricsReader {
    private NamedPipeClientStream _pipe;
    private StreamReader           _reader;
    
    public event Action<TrainingMetrics> OnMetrics;
    
    public async Task StartAsync(string pipeName) {
        _pipe = new NamedPipeClientStream(".", pipeName, 
                    PipeDirection.In, PipeOptions.Asynchronous);
        await _pipe.ConnectAsync();
        _reader = new StreamReader(_pipe);
        
        while (true) {
            string? line = await _reader.ReadLineAsync();
            if (line == null) break;
            var metrics = JsonSerializer.Deserialize<TrainingMetrics>(line);
            OnMetrics?.Invoke(metrics);
        }
    }
}
```

---

## GUI Layout

```
┌─────────────────────────────────────────────────────────┐
│  ModernLLM Training Monitor              [Pause] [Stop] │
├──────────────┬──────────────┬────────────────────────────┤
│ Step: 24,500 │ Loss: 2.341  │ LR: 3.00e-4               │
│ Tokens: 5.2B │ Perp: 10.2   │ Tok/sec: 142,000          │
│ MFU: 43.2%   │ Grad: 0.82   │ GPU Mem: 18.4 GB          │
├──────────────┴──────────────┴────────────────────────────┤
│                   LOSS CURVE                             │
│  3.5 ┤                                                   │
│  3.0 ┤  ╲                                                │
│  2.5 ┤   ╲──╲                                            │
│  2.0 ┤      ╲────────────────                            │
│      └────────────────────────────────────────── steps  │
├─────────────────────────────────────────────────────────┤
│                 TOKENS/SEC & MFU                         │
│  (dual-axis line chart)                                  │
├──────────────────────────┬──────────────────────────────┤
│     CHECKPOINT MANAGER   │      MoE EXPERT LOAD         │
│  ✓ step_10000.bin  [Load]│  ████████░░░░░░░░  Expert 1  │
│  ✓ step_20000.bin  [Load]│  ██████████░░░░░░  Expert 2  │
│  ✓ step_24500.bin  [Load]│  ██████░░░░░░░░░░  Expert 3  │
│     [Save Now]           │  ... (load balance heatmap)  │
└──────────────────────────┴──────────────────────────────┘
```

---

## WPF Implementation

### Main Window (MVVM pattern)

```csharp
// MainViewModel.cs
public class MainViewModel : INotifyPropertyChanged {
    
    // Observable collections for charts
    public ObservableCollection<DataPoint> LossHistory    { get; } = new();
    public ObservableCollection<DataPoint> PerplexityHistory { get; } = new();
    public ObservableCollection<DataPoint> ThroughputHistory { get; } = new();
    public ObservableCollection<ExpertLoad> ExpertLoads   { get; } = new();
    
    // Current stats (bound to UI)
    private float _currentLoss;
    public float CurrentLoss {
        get => _currentLoss;
        set { _currentLoss = value; OnPropertyChanged(); }
    }
    
    private long _tokensSeen;
    public string TokensSeen => FormatBillions(_tokensSeen);
    
    // Commands
    public ICommand PauseCommand  { get; }
    public ICommand StopCommand   { get; }
    public ICommand SaveNowCommand { get; }
    public ICommand LoadCheckpointCommand { get; }
    
    // Called when new metrics arrive from C++ pipe
    public void UpdateMetrics(TrainingMetrics m) {
        App.Current.Dispatcher.Invoke(() => {
            CurrentLoss = m.Loss;
            _tokensSeen = m.TokensSeen;
            
            LossHistory.Add(new DataPoint(m.Step, m.Loss));
            ThroughputHistory.Add(new DataPoint(m.Step, m.TokensPerSec));
            
            // Keep last 1000 points in chart
            if (LossHistory.Count > 1000) LossHistory.RemoveAt(0);
        });
    }
}
```

### Charting

Use **LiveCharts2** (free, .NET 8 compatible, smooth animation):

```xml
<!-- MainWindow.xaml -->
<lvc:CartesianChart Series="{Binding LossSeries}"
                    XAxes="{Binding StepAxis}"
                    YAxes="{Binding LossAxis}"
                    AnimationsSpeed="00:00:00.100"
                    EasingFunction="{x:Null}"/>
```

```csharp
// Chart series config
public ISeries[] LossSeries => new ISeries[] {
    new LineSeries<DataPoint> {
        Values = LossHistory,
        Stroke = new SolidColorPaint(SKColors.CornflowerBlue, 2),
        Fill = null,
        GeometrySize = 0,
        LineSmoothness = 0.2
    }
};
```

### Checkpoint Manager Panel

```csharp
// CheckpointManager.cs
public class CheckpointManager {
    private string _checkpointDir;
    
    public ObservableCollection<CheckpointInfo> Checkpoints { get; } = new();
    
    public void Scan() {
        var files = Directory.GetFiles(_checkpointDir, "*.bin")
                             .OrderByDescending(f => File.GetLastWriteTime(f));
        Checkpoints.Clear();
        foreach (var f in files) {
            var info = LoadCheckpointMeta(f);
            Checkpoints.Add(info);
        }
    }
    
    // Send "load checkpoint" command to C++ via control pipe
    public void LoadCheckpoint(string path) {
        SendCommand(new { command = "load_checkpoint", path });
    }
    
    public void SaveNow() {
        SendCommand(new { command = "save_checkpoint" });
    }
}

public record CheckpointInfo(
    string Path,
    int Step,
    float Loss,
    long TokensSeen,
    DateTime SavedAt
);
```

### MoE Expert Load Heatmap

Visualize if MoE experts are balanced (important — indicates training health):

```csharp
public class ExpertLoadHeatmap : UserControl {
    // Renders a grid of colored cells
    // Green = balanced load, Red = overloaded expert
    // Data comes from aux_loss and per-expert token counts in metrics
    
    public void Update(float[] expertLoads) {
        // Normalize 0..1
        // Draw colored rectangles: green (0.015) → red (0.05+)
    }
}
```

---

## Control Channel (C++ ← C#)

The C# GUI can also send commands to C++:

```csharp
// Commands C# sends to C++
public enum TrainerCommand {
    Pause,
    Resume,
    Stop,
    SaveCheckpoint,
    LoadCheckpoint,
    SetLR,          // adjust LR on the fly
    SetBatchSize
}

// Sent as JSON over a second named pipe (control pipe)
{ "command": "pause" }
{ "command": "set_lr", "value": 1e-4 }
{ "command": "load_checkpoint", "path": "checkpoints/step_20000.bin" }
```

C++ trainer reads from control pipe in a background thread, applies commands between batches (never mid-step).

---

## Checkpoint File Format

Binary format for fast save/load:

```
[HEADER]
magic: u32 = 0x4C4C4D58 ("LLMX")
version: u32 = 1
step: u64
tokens_seen: u64
loss: f32
lr: f32
config_len: u32
config_json: char[config_len]   ← model config JSON

[WEIGHTS]
For each weight tensor:
  name_len: u32
  name: char[name_len]
  dtype: u8  (0=fp32, 1=bf16)
  ndim: u32
  shape: u32[ndim]
  data: byte[numel * sizeof(dtype)]

[OPTIMIZER]
For each param:
  m_state: f32[numel]   (or int8 if 8-bit optimizer)
  v_state: f32[numel]
```

**Auto-save triggers:**
- Every N steps (configurable, default 1000)
- On C# "Save Now" button
- On graceful shutdown (SIGINT/SIGTERM handler)
- On detected training instability (loss spike >2× recent avg)

---

## NuGet Packages

```xml
<PackageReference Include="LiveChartsCore.SkiaSharpView.WPF" Version="2.0.0-rc3" />
<PackageReference Include="CommunityToolkit.Mvvm"            Version="8.3.2" />
<PackageReference Include="Microsoft.Extensions.Hosting"     Version="8.0.0" />
```

---

## Metrics Logged to Disk

Every metric emission is also appended to `training_log.jsonl` for post-analysis:

```jsonl
{"step":1000,"loss":3.241,"lr":3e-4,"tok_per_sec":141200,...}
{"step":2000,"loss":2.891,"lr":3e-4,"tok_per_sec":143100,...}
```

This enables offline analysis, plotting in Python/matplotlib, and comparison across runs.
