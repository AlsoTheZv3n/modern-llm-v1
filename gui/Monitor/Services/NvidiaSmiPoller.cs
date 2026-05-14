using System;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

using ModernLLM.Monitor.Models;

namespace ModernLLM.Monitor.Services;

/// <summary>
/// Polls nvidia-smi on a fixed interval. Emits a GpuStats event per successful
/// query; emits an error message on failure (e.g. nvidia-smi not on PATH).
/// </summary>
public sealed class NvidiaSmiPoller : IDisposable
{
    private readonly TimeSpan _interval;
    private CancellationTokenSource? _cts;
    private Task? _loop;

    public event Action<GpuStats>? OnSample;
    public event Action<string>? OnError;

    public NvidiaSmiPoller(double intervalSec = 2.0)
    {
        _interval = TimeSpan.FromSeconds(intervalSec);
    }

    public void Start()
    {
        if (_loop is not null) return;
        _cts = new CancellationTokenSource();
        _loop = Task.Run(() => Loop(_cts.Token));
    }

    private async Task Loop(CancellationToken ct)
    {
        bool reportedError = false;
        while (!ct.IsCancellationRequested)
        {
            try
            {
                var s = QueryOnce();
                if (s is not null)
                {
                    reportedError = false;
                    OnSample?.Invoke(s);
                }
            }
            catch (Exception ex)
            {
                if (!reportedError)
                {
                    OnError?.Invoke($"nvidia-smi: {ex.Message}");
                    reportedError = true;
                }
            }

            try { await Task.Delay(_interval, ct); } catch { /* shutdown */ }
        }
    }

    /// <summary>
    /// Run nvidia-smi once. Returns null if the call succeeded but parse
    /// failed; throws on process-level failure (nvidia-smi missing / error
    /// exit). One sample = first GPU only.
    /// </summary>
    private static GpuStats? QueryOnce()
    {
        var psi = new ProcessStartInfo
        {
            FileName = "nvidia-smi",
            Arguments = "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw " +
                         "--format=csv,noheader,nounits",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };

        using var p = Process.Start(psi)
            ?? throw new InvalidOperationException("could not start nvidia-smi process");

        string output = p.StandardOutput.ReadToEnd();
        p.WaitForExit(3000);
        if (p.ExitCode != 0)
            throw new InvalidOperationException(
                $"nvidia-smi exited {p.ExitCode}: {p.StandardError.ReadToEnd().Trim()}");

        var firstLine = output.Split('\n')
                              .Select(l => l.Trim())
                              .FirstOrDefault(l => l.Length > 0);
        if (string.IsNullOrEmpty(firstLine)) return null;

        var parts = firstLine.Split(',').Select(x => x.Trim()).ToArray();
        if (parts.Length < 5) return null;

        // Some fields can be "[N/A]" on integrated/old GPUs; tolerate that.
        int parseIntOr(string s, int fallback) =>
            int.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out var v) ? v : fallback;
        double parseDoubleOr(string s, double fallback) =>
            double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out var v) ? v : fallback;

        return new GpuStats(
            UtilPct: parseIntOr(parts[0], 0),
            VramUsedMB: parseIntOr(parts[1], 0),
            VramTotalMB: parseIntOr(parts[2], 0),
            TempC: parseIntOr(parts[3], 0),
            PowerW: parseDoubleOr(parts[4], 0.0)
        );
    }

    public void Dispose()
    {
        _cts?.Cancel();
        _cts?.Dispose();
        _cts = null;
        _loop = null;
    }
}
