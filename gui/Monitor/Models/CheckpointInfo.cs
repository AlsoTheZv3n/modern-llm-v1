using System;

namespace ModernLLM.Monitor.Models;

/// <summary>One *.ckpt file discovered under runs/.</summary>
public sealed record CheckpointInfo(
    string Path,
    string FileName,
    long SizeBytes,
    DateTime LastModified
)
{
    public double SizeMB => SizeBytes / (1024.0 * 1024.0);

    /// <summary>"3 minutes ago" / "2 days ago" / "just now".</summary>
    public string Age
    {
        get
        {
            var dt = DateTime.UtcNow - LastModified.ToUniversalTime();
            if (dt.TotalSeconds < 30) return "just now";
            if (dt.TotalMinutes < 1) return $"{dt.Seconds}s ago";
            if (dt.TotalHours < 1) return $"{dt.Minutes}m ago";
            if (dt.TotalDays < 1) return $"{(int)dt.TotalHours}h ago";
            return $"{(int)dt.TotalDays}d ago";
        }
    }

    /// <summary>"123.4 MB".</summary>
    public string SizeDisplay => $"{SizeMB,7:F1} MB";

    /// <summary>"finewebedu_50m.ckpt    123.4 MB    3m ago".</summary>
    public string Display => $"{FileName}    {SizeDisplay}    {Age}";
}
