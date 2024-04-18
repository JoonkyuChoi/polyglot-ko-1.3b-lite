REM ----------------------------------------------------------------------------
REM .create-symbolic-links.bat
REM find symlink > dir /S /AL
REM ----------------------------------------------------------------------------
REM delete symbolic-links
REM ------------------------------------
RMDIR /Q ".\resources"
REM ------------------------------------
REM create symbolic-links
REM ------------------------------------
mklink /D ".\resources"   "..\..\..\..\..\..\..\AI-LFS\SLM"
REM ------------------------------------
REM exclude symbolic-links for trace : .git/index
REM ------------------------------------
git update-index --assume-unchanged "./resources"
REM ----------------------------------------------------------------------------
