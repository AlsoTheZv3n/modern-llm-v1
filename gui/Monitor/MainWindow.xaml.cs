using System;
using System.IO;
using System.Linq;
using System.Windows;

using Microsoft.Win32;

using ModernLLM.Monitor.Services;
using ModernLLM.Monitor.ViewModels;
using ModernLLM.Monitor.Views;

namespace ModernLLM.Monitor;

public partial class MainWindow : Window
{
    private readonly MainViewModel _vm;

    public MainWindow()
    {
        InitializeComponent();
        _vm = (MainViewModel)DataContext;
        Closing += (_, _) => _vm.Dispose();

        // Auto-discover the most recent training log under runs/, walking up
        // from the binary directory because debug builds live deep under
        // bin/Debug/...
        var auto = TryFindMostRecentLog();
        if (auto != null) _vm.OpenFile(auto);
    }

    private static string? TryFindMostRecentLog()
    {
        var dir = AppContext.BaseDirectory;
        for (int i = 0; i < 8; i++)
        {
            var runs = Path.Combine(dir, "runs");
            if (Directory.Exists(runs))
            {
                var newest = Directory.GetFiles(runs, "*.jsonl", SearchOption.TopDirectoryOnly)
                                       .Select(p => new FileInfo(p))
                                       .OrderByDescending(f => f.LastWriteTimeUtc)
                                       .FirstOrDefault();
                if (newest != null) return newest.FullName;
            }
            var parent = Directory.GetParent(dir);
            if (parent == null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private void OpenLog_Click(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog
        {
            Filter = "Training log (*.jsonl)|*.jsonl|All files (*.*)|*.*",
            Title = "Choose a training JSONL log"
        };
        if (dlg.ShowDialog() == true)
        {
            _vm.OpenFile(dlg.FileName);
        }
    }

    private void Reload_Click(object sender, RoutedEventArgs e)
    {
        if (!string.IsNullOrEmpty(_vm.CurrentFile) && _vm.CurrentFile != "(no file)")
        {
            string p = _vm.CurrentFile;
            _vm.CloseFile();
            _vm.OpenFile(p);
        }
    }

    private void OpenSampler_Click(object sender, RoutedEventArgs e)
    {
        var repoRoot = CheckpointBrowser.FindRepoRoot(AppContext.BaseDirectory)
                       ?? AppContext.BaseDirectory;
        var samplerVm = new SamplerViewModel(repoRoot);
        var win = new SamplerWindow(samplerVm) { Owner = this };
        win.Show();
    }
}
