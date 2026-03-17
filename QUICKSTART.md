# Quick Start: Connecting Visual Studio Code to catfda

This guide answers the question "how to connect Visual Studio?" for the catfda R package.

## Quick Setup (5 minutes)

### 1. Install Prerequisites
- [Visual Studio Code](https://code.visualstudio.com/)
- [R (latest version)](https://www.r-project.org/)

### 2. Install VS Code Extensions
Open VS Code → Extensions (Ctrl+Shift+X) → Search and install:
- **R Extension for Visual Studio Code** (by REditorSupport)

### 3. Open the Project
- Download this repository or clone it
- Open the folder in VS Code: `File → Open Folder`
- Or open the workspace: `File → Open Workspace from File` → select `catfda.code-workspace`

### 4. Install catfda Package
Open VS Code terminal (`Ctrl+`` `) and run:
```r
# Start R
R

# Install the package
install.packages("devtools")
devtools::install_github("XiaoxiaChampon/catfda")

# Test it works
library(catfda)
```

### 5. Start Coding
- Create new R file: `File → New File` → Save as `test.R`
- Copy sample code from README or `R/cluster_sample.R`
- Run code with `Ctrl+Enter`

## That's it! 
You're now connected and ready to use catfda with Visual Studio Code.

For detailed setup instructions, see [VISUAL_STUDIO_SETUP.md](VISUAL_STUDIO_SETUP.md).