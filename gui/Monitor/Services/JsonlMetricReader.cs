using System;
using System.IO;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

using ModernLLM.Monitor.Models;

namespace ModernLLM.Monitor.Services;

/// <summary>
/// Tails a JSONL log file. Reads everything that's already there on Start(),
/// then polls for new bytes and emits one event per new line.
///
/// Polling (instead of FileSystemWatcher) because the trainer rewrites the
/// file when started without --resume, which races with Watcher events on
/// some Windows configurations. A 200 ms polling loop is plenty responsive
/// for a training log that emits at most a few lines per second.
/// </summary>
public sealed class JsonlMetricReader : IDisposable
{
    private readonly string _path;
    private readonly TimeSpan _pollInterval;
    private CancellationTokenSource? _cts;
    private Task? _loop;
    private long _lastSize;

    public event Action<MetricSample>? OnSample;
    public event Action? OnReset;     // file was truncated / replaced

    public JsonlMetricReader(string path, TimeSpan? pollInterval = null)
    {
        _path = path;
        _pollInterval = pollInterval ?? TimeSpan.FromMilliseconds(200);
    }

    public string Path => _path;

    public void Start()
    {
        Stop();
        _cts = new CancellationTokenSource();
        _loop = Task.Run(() => Loop(_cts.Token));
    }

    public void Stop()
    {
        _cts?.Cancel();
        try { _loop?.Wait(500); } catch { /* ignore */ }
        _cts?.Dispose();
        _cts = null;
        _loop = null;
        _lastSize = 0;
    }

    public void Dispose() => Stop();

    private async Task Loop(CancellationToken ct)
    {
        long pos = 0;
        while (!ct.IsCancellationRequested)
        {
            try
            {
                if (File.Exists(_path))
                {
                    var info = new FileInfo(_path);
                    long size = info.Length;
                    if (size < pos)
                    {
                        // Truncated — file was rewritten. Reset.
                        pos = 0;
                        OnReset?.Invoke();
                    }
                    if (size > pos)
                    {
                        // Read everything from pos to size.
                        using var fs = new FileStream(
                            _path, FileMode.Open, FileAccess.Read,
                            FileShare.ReadWrite | FileShare.Delete);
                        fs.Seek(pos, SeekOrigin.Begin);
                        using var sr = new StreamReader(fs);
                        string? line;
                        while ((line = await sr.ReadLineAsync(ct)
                                                .ConfigureAwait(false)) != null)
                        {
                            if (line.Length == 0) continue;
                            if (TryParseSample(line, out var s))
                            {
                                OnSample?.Invoke(s);
                            }
                        }
                        pos = fs.Position;
                    }
                    _lastSize = size;
                }
                else
                {
                    pos = 0;
                }
            }
            catch (OperationCanceledException) { break; }
            catch (IOException)
            {
                // file might be locked momentarily; just retry next tick
            }

            try { await Task.Delay(_pollInterval, ct).ConfigureAwait(false); }
            catch (OperationCanceledException) { break; }
        }
    }

    private static double GetDouble(JsonElement obj, string name, double fallback)
    {
        if (obj.TryGetProperty(name, out var v) &&
            (v.ValueKind == JsonValueKind.Number))
        {
            return v.GetDouble();
        }
        return fallback;
    }

    private static long GetLong(JsonElement obj, string name, long fallback)
    {
        if (obj.TryGetProperty(name, out var v) &&
            (v.ValueKind == JsonValueKind.Number))
        {
            return v.GetInt64();
        }
        return fallback;
    }

    private static bool TryParseSample(string line, out MetricSample s)
    {
        s = null!;
        try
        {
            using var doc = JsonDocument.Parse(line);
            var root = doc.RootElement;
            if (root.ValueKind != JsonValueKind.Object) return false;

            int step = (int)GetLong(root, "step", 0);
            double loss = GetDouble(root, "loss", double.NaN);
            double valLoss = GetDouble(root, "val_loss", double.NaN);
            double lr = GetDouble(root, "lr", double.NaN);
            double grad = GetDouble(root, "grad_norm", double.NaN);
            long tokens = GetLong(root, "tokens_seen", 0);
            double elapsed = GetDouble(root, "elapsed_sec", 0.0);

            s = new MetricSample(step, loss, valLoss, lr, grad, tokens, elapsed);
            return true;
        }
        catch
        {
            return false;
        }
    }
}
