## R CMD check results

0 errors | 0 warnings | 1 note

    * checking for future file timestamps ... NOTE
      unable to verify current time

This NOTE is a known false positive: the check environment could not reach a
time server to verify timestamps. It does not reflect any problem in the
package.

## Test environments

* Local: Windows 11, R 4.4.1 (ucrt)
* win-builder: R-devel, R-release
* R-hub: Ubuntu 22.04 (R-devel), macOS (R-release)

## Downstream dependencies

None. This is a new submission.
