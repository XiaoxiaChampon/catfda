# Using catfda with Visual Studio Code

This document provides instructions on how to set up and use the `catfda` R package with Visual Studio Code.

## Prerequisites

- Visual Studio Code installed
- R (version 4.0 or higher) installed on your system
- Git (for package installation from GitHub)

## Setup Instructions

### 1. Install Visual Studio Code Extensions

Install the following essential extensions for R development:

1. **R Extension for Visual Studio Code** (by REditorSupport)
   - Provides R language support, syntax highlighting, and debugging
   - Search for "R" in the VS Code Extensions marketplace

2. **R Debugger** (by RDebugger)
   - Enables debugging R code within VS Code

3. **R Tools** (optional, by Mikhail Arkhipov)
   - Additional R development tools

### 2. Configure R Language Server

The R extension uses the R Language Server for enhanced functionality. Install it in R:

```r
# Install the R Language Server
install.packages("languageserver")
```

### 3. Configure VS Code Settings

Add the following to your VS Code `settings.json` (File → Preferences → Settings → Open Settings JSON):

```json
{
    "r.rterm.windows": "C:\\Program Files\\R\\R-4.x.x\\bin\\R.exe",
    "r.rterm.mac": "/usr/local/bin/R",
    "r.rterm.linux": "/usr/bin/R",
    "r.rpath.windows": "C:\\Program Files\\R\\R-4.x.x\\bin\\R.exe",
    "r.rpath.mac": "/usr/local/bin/R",
    "r.rpath.linux": "/usr/bin/R",
    "r.plot.useHttpgd": true,
    "r.bracketedPaste": true,
    "r.sessionWatcher": true
}
```

*Note: Adjust the R paths according to your R installation location.*

### 4. Install catfda Package

Open the integrated terminal in VS Code (`Ctrl+`` ` or `View → Terminal`) and run R:

```r
# Install devtools if not already installed
if (!requireNamespace("devtools", quietly = TRUE)) {
    install.packages("devtools")
}

# Install catfda from GitHub
library(devtools)
install_github("XiaoxiaChampon/catfda")

# Load the package
library(catfda)
```

### 5. Verify Installation

Create a new R file (`Ctrl+N`, save as `.R`) and test the installation:

```r
# Load the catfda package
library(catfda)

# Test with sample data (as mentioned in README)
sample_data <- matrix(sample(c(0,1,2), 100*250, replace=TRUE), nrow=100, ncol=250)
time_points <- seq(0, 1, length=250)

# Run clustering analysis
catfdclust <- catfdcluster(
    sample_data, 
    time_points, 
    25, 3, 3, 0.9, 4, 5, 2, 
    "happ", "two"
)

# Check results
print("catfda package successfully installed and working!")
```

## Working with catfda in VS Code

### Running R Code

1. **Interactive Mode**: Use `Ctrl+Enter` to send current line or selection to R terminal
2. **Run All**: Use `Ctrl+Shift+S` to source the entire file
3. **R Terminal**: Open R terminal with `Ctrl+Shift+P` → "R: Create R Terminal"

### Key Features

- **Syntax Highlighting**: R code is automatically highlighted
- **IntelliSense**: Auto-completion for R functions and variables
- **Function Help**: Hover over functions to see documentation
- **Debugging**: Set breakpoints and debug R code interactively
- **Plot Viewer**: View R plots in VS Code or external viewer

### Useful Keyboard Shortcuts

- `Ctrl+Enter`: Run current line/selection
- `Ctrl+Shift+Enter`: Run current chunk (if using R Markdown)
- `Ctrl+Shift+P`: Command palette
- `Ctrl+Shift+C`: Toggle line comment
- `F1`: Show function help

## Sample Workflow

1. Create a new R file: `File → New File` and save with `.R` extension
2. Set up your analysis:
   ```r
   library(catfda)
   # Your analysis code here
   ```
3. Run code interactively with `Ctrl+Enter`
4. Use the integrated terminal for package management and R console operations
5. Save plots and results using R's standard functions

## Troubleshooting

### Common Issues

1. **R not found**: Ensure R is installed and the path is correctly set in VS Code settings
2. **Package not loading**: Check that all dependencies are installed:
   ```r
   install.packages(c("fda", "refund", "mgcv", "funData", "MFPCA", 
                      "dbscan", "fossil", "NbClust", "ggplot2"))
   ```
3. **Language server not working**: Restart VS Code and ensure `languageserver` package is installed

### Getting Help

- Use `?function_name` in R console for function documentation
- Check the [GitHub repository](https://github.com/XiaoxiaChampon/catfda) for issues and documentation
- Use `help(package="catfda")` for package overview

## Additional Resources

- [R Extension Documentation](https://github.com/REditorSupport/vscode-R)
- [VS Code R Tutorial](https://code.visualstudio.com/docs/languages/r)
- [catfda GitHub Repository](https://github.com/XiaoxiaChampon/catfda)