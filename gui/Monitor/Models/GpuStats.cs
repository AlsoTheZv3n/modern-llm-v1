namespace ModernLLM.Monitor.Models;

/// <summary>
/// One nvidia-smi sample. All fields are the raw values from --format=csv,noheader,nounits.
/// </summary>
public sealed record GpuStats(
    int UtilPct,
    int VramUsedMB,
    int VramTotalMB,
    int TempC,
    double PowerW
);
