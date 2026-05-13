using System;
using System.IO;
using System.Windows;

using Microsoft.Win32;

using ModernLLM.Monitor.ViewModels;

namespace ModernLLM.Monitor;

public partial class MainWindow : Window
{
    private readonly MainViewModel _vm;

    public MainWindow()
    {
        InitializeComponent();
        _vm = (MainViewModel)DataContext;

        // Auto-discover the modern_gpt log if it exists at the canonical path
        // (relative to repo root). We walk up a few levels from the binary
        // directory because debug builds live deep under bin/Debug/...
        var auto = TryFindDefaultLog();
        if (auto != null) _vm.OpenFile(auto);
    }

    private static string? TryFindDefaultLog()
    {
        var dir = AppContext.BaseDirectory;
        for (int i = 0; i < 8; i++)
        {
            var candidate = Path.Combine(dir, "runs", "modern_gpt_log.jsonl");
            if (File.Exists(candidate)) return candidate;
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
}
