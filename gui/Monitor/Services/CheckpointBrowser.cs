using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

using ModernLLM.Monitor.Models;

namespace ModernLLM.Monitor.Services;

/// <summary>Lists *.ckpt files under a given directory, newest first.</summary>
public static class CheckpointBrowser
{
    public static IReadOnlyList<CheckpointInfo> Discover(string runsDir)
    {
        if (!Directory.Exists(runsDir)) return Array.Empty<CheckpointInfo>();

        return Directory.GetFiles(runsDir, "*.ckpt", SearchOption.TopDirectoryOnly)
                         .Select(p =>
                         {
                             var fi = new FileInfo(p);
                             return new CheckpointInfo(
                                 Path: fi.FullName,
                                 FileName: fi.Name,
                                 SizeBytes: fi.Length,
                                 LastModified: fi.LastWriteTime
                             );
                         })
                         .OrderByDescending(c => c.LastModified)
                         .ToList();
    }

    /// <summary>
    /// Walks up from `startDir` looking for a directory containing `runs/`.
    /// Returns the absolute path to the runs/ folder, or null if not found
    /// within the parent chain.
    /// </summary>
    public static string? FindRunsDir(string startDir, int maxLevels = 8)
    {
        var dir = startDir;
        for (int i = 0; i < maxLevels; i++)
        {
            var candidate = Path.Combine(dir, "runs");
            if (Directory.Exists(candidate)) return candidate;
            var parent = Directory.GetParent(dir);
            if (parent == null) return null;
            dir = parent.FullName;
        }
        return null;
    }

    /// <summary>
    /// Walks up from `startDir` looking for a directory containing both `llm/`
    /// (the C++ engine) and `scripts/` (the Python wrappers). Returns the
    /// repo root, or null if not found.
    /// </summary>
    public static string? FindRepoRoot(string startDir, int maxLevels = 8)
    {
        var dir = startDir;
        for (int i = 0; i < maxLevels; i++)
        {
            if (Directory.Exists(Path.Combine(dir, "llm")) &&
                Directory.Exists(Path.Combine(dir, "scripts")))
                return dir;
            var parent = Directory.GetParent(dir);
            if (parent == null) return null;
            dir = parent.FullName;
        }
        return null;
    }
}
