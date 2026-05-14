using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace ModernLLM.Monitor.Services;

/// <summary>
/// Model-config bundle for sampling. Mirrors the args expected by
/// scripts\sample.py.
/// </summary>
public sealed record SampleConfig(
    string CheckpointPath,
    string MetaPath,
    string Prompt,
    int NumTokens,
    int SeqLen,
    int DModel,
    int NHeads,
    int NKvHeads,
    int NLayers,
    int DFfn,
    double Temperature,
    int Seed
);

public sealed record SampleResult(
    bool Success,
    string Text,
    string Stderr,
    int ExitCode
);

/// <summary>
/// Spawns scripts\sample.py and captures its output. Runs on a worker thread
/// so the UI stays responsive.
/// </summary>
public static class SamplerService
{
    public static async Task<SampleResult> RunAsync(
        string repoRoot, SampleConfig cfg, CancellationToken ct = default)
    {
        var script = Path.Combine(repoRoot, "scripts", "sample.py");
        if (!File.Exists(script))
            return new SampleResult(false, "", $"sample.py not found at {script}", -1);

        var args = new[]
        {
            "\"" + script + "\"",
            "--ckpt", "\"" + cfg.CheckpointPath + "\"",
            "--meta", "\"" + cfg.MetaPath + "\"",
            "--prompt", "\"" + cfg.Prompt.Replace("\"", "\\\"") + "\"",
            "--num", cfg.NumTokens.ToString(),
            "--seq-len", cfg.SeqLen.ToString(),
            "--d-model", cfg.DModel.ToString(),
            "--n-heads", cfg.NHeads.ToString(),
            "--n-kv-heads", cfg.NKvHeads.ToString(),
            "--n-layers", cfg.NLayers.ToString(),
            "--d-ffn", cfg.DFfn.ToString(),
            "--temp", cfg.Temperature.ToString(System.Globalization.CultureInfo.InvariantCulture),
            "--seed", cfg.Seed.ToString(),
        };
        var psi = new ProcessStartInfo
        {
            FileName = "python",
            Arguments = string.Join(" ", args),
            WorkingDirectory = repoRoot,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
            StandardOutputEncoding = System.Text.Encoding.UTF8,
            StandardErrorEncoding = System.Text.Encoding.UTF8,
        };

        try
        {
            using var p = Process.Start(psi)
                ?? throw new InvalidOperationException("could not start python");

            var stdoutTask = p.StandardOutput.ReadToEndAsync();
            var stderrTask = p.StandardError.ReadToEndAsync();

            // Wait for process or cancellation
            var exitTcs = new TaskCompletionSource();
            p.EnableRaisingEvents = true;
            p.Exited += (_, _) => exitTcs.TrySetResult();

            using (ct.Register(() => { try { if (!p.HasExited) p.Kill(true); } catch { } }))
            {
                await exitTcs.Task.WaitAsync(ct).ConfigureAwait(false);
            }

            string stdout = await stdoutTask.ConfigureAwait(false);
            string stderr = await stderrTask.ConfigureAwait(false);

            return new SampleResult(
                Success: p.ExitCode == 0,
                Text: stdout,
                Stderr: stderr,
                ExitCode: p.ExitCode);
        }
        catch (OperationCanceledException)
        {
            return new SampleResult(false, "", "(cancelled)", -1);
        }
        catch (Exception ex)
        {
            return new SampleResult(false, "", $"failed to run python: {ex.Message}", -1);
        }
    }
}
