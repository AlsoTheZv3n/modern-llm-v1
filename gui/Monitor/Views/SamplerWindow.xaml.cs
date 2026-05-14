using System.Windows;

using ModernLLM.Monitor.ViewModels;

namespace ModernLLM.Monitor.Views;

public partial class SamplerWindow : Window
{
    public SamplerWindow(SamplerViewModel vm)
    {
        InitializeComponent();
        DataContext = vm;
    }
}
