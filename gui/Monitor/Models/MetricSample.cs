namespace ModernLLM.Monitor.Models;

/// <summary>
/// One JSONL log line emitted by the trainer.
/// All numeric fields default to NaN so we can tell which were absent.
/// </summary>
public sealed record MetricSample(
    int Step,
    double Loss,
    double ValLoss,
    double Lr,
    double GradNorm,
    long TokensSeen,
    double ElapsedSec
);
